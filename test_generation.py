"""
Test generation for PetGPT (with optional SMPL conditioning).

Generates pet motion on ALL test-set sequences, computes FID using
a pretrained PetMotionClassifier, and optionally saves per-file outputs.

Usage:
  python test_generation.py --config configs/pet_gpt_smpl.yaml --gpt_ckpt checkpoints/pet_gpt_smpl_best.pt
"""
import os
import sys
import json
import re
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

from models.pet_gpt import PetGPT
from models.pet_motion_classifier import PetMotionClassifier
from models.wan_mano_vqvae import WanManoVQVAE
from models.wan_pet_vqvae import WanPetVQVAE
from models.wan_smpl_vqvae import WanSmplVQVAE
from utils.pet_fid import compute_pet_fid


def load_checkpoint(path, device):
    """Load a public weights-only checkpoint, with trusted legacy fallback.

    Public release checkpoints contain only tensors, primitive metadata, and
    plain dictionaries, so they load with ``weights_only=True``.  The fallback
    keeps locally produced pre-release checkpoints usable during migration;
    it must only be used with checkpoints from a trusted source.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as weights_only_error:
        print(f'Warning: {path} is a legacy pickle checkpoint; only load it '
              f'if you trust its source ({type(weights_only_error).__name__}).')
        return torch.load(path, map_location=device, weights_only=False)


def load_pretrained_vqvae(ckpt_path, device, model_cls=WanPetVQVAE,
                           **kwargs):
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = EasyDict(ckpt['config'])
    hps = EasyDict(cfg.structure) if hasattr(cfg, 'structure') else cfg
    model = model_cls(hps, **kwargs).to(device)
    state = ckpt['model']
    if any(key.startswith('module.') for key in state):
        state = {key.replace('module.', ''): value
                 for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_pretrained_mano_vqvae(ckpt_path, device):
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = EasyDict(ckpt['config'])
    structure = cfg.structure if hasattr(cfg, 'structure') else cfg
    model = WanManoVQVAE(
        EasyDict(structure.left), EasyDict(structure.right)).to(device)
    state = ckpt['model']
    if any(key.startswith('module.') for key in state):
        state = {key.replace('module.', ''): value
                 for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_norm_stats(pet_root, mano_root=None, smpl_root=None):
    """Load normalization stats for pet, pet_rel, mano, and optionally smpl."""
    with open(os.path.join(pet_root, 'norm_stats.json')) as f:
        pet_d = json.load(f)
    pet_stats = {
        'mean': np.array(pet_d['mean'], dtype=np.float32),
        'std': np.array(pet_d['std'], dtype=np.float32),
    }
    if 'pet_rel_mean' in pet_d:
        pet_stats['pet_rel_mean'] = np.array(pet_d['pet_rel_mean'],
                                              dtype=np.float32)
        pet_stats['pet_rel_std'] = np.array(pet_d['pet_rel_std'],
                                             dtype=np.float32)

    mano_stats = None
    if mano_root is not None:
        with open(os.path.join(mano_root, 'mano_norm_stats.json')) as f:
            mano_d = json.load(f)
        mano_stats = {}
        for key in ('left_pos', 'left_rot', 'right_pos', 'right_rot'):
            mano_stats[key] = (
                np.array(mano_d[f'{key}_mean'], dtype=np.float32),
                np.array(mano_d[f'{key}_std'], dtype=np.float32),
            )

    smpl_stats = None
    if smpl_root is not None:
        with open(os.path.join(smpl_root, 'smpl_norm_stats.json')) as f:
            smpl_d = json.load(f)
        smpl_stats = {}
        for key in ('pos', 'rot'):
            smpl_stats[key] = (
                np.array(smpl_d[f'{key}_mean'], dtype=np.float32),
                np.array(smpl_d[f'{key}_std'], dtype=np.float32),
            )

    return pet_stats, mano_stats, smpl_stats


def get_test_files(pet_root, mano_root=None, smpl_root=None, audio_root=None,
                   test_ratio=0.2, split_seed=42, split='val'):
    """Get split filenames using per-dog split (same as training).

    Args:
        split: 'val'/'test' (default evaluation split) or 'train'
    """
    from collections import defaultdict
    pet_files = set(f for f in os.listdir(pet_root) if f.endswith('.npy'))
    if mano_root is not None:
        mano_files = set(f for f in os.listdir(mano_root) if f.endswith('.npy'))
        common = sorted(pet_files & mano_files)
    else:
        common = sorted(pet_files)
    if smpl_root is not None:
        smpl_files = set(f for f in os.listdir(smpl_root) if f.endswith('.npy'))
        common = sorted(set(common) & smpl_files)
    if audio_root is not None:
        audio_files = set(f for f in os.listdir(audio_root) if f.endswith('.npy'))
        common = sorted(set(common) & audio_files)

    rng = np.random.RandomState(split_seed)
    dog_to_files = defaultdict(list)
    for fn in common:
        m = re.search(r'(dog\d+)', fn)
        dog_to_files[m.group(1) if m else '__unk__'].append(fn)

    train_files, val_files = [], []
    for did in sorted(dog_to_files):
        files = sorted(dog_to_files[did])
        n_val = max(1, int(round(len(files) * test_ratio)))
        idx = rng.permutation(len(files))
        val_idx = set(idx[:n_val].tolist())
        for i, f in enumerate(files):
            if i in val_idx:
                val_files.append(f)
            else:
                train_files.append(f)
    chosen = train_files if split == 'train' else val_files
    return sorted(chosen)


def preprocess_sequence(pet_path, mano_path, pet_stats, mano_stats,
                        seq_len=300, smpl_path=None, smpl_stats=None,
                        smpl_camera='aria01',
                        audio_path=None, audio_fps=75, motion_fps=30):
    """Load and preprocess a paired pet/mano(/smpl/audio) sequence into chunks."""
    pet_raw = np.load(pet_path).astype(np.float32)
    T_pet = pet_raw.shape[0]
    T = T_pet

    mano_data = None
    if mano_path is not None:
        mano_data = np.load(mano_path, allow_pickle=True).item()
        T_mano = mano_data['left']['joints'].shape[0]
        T = min(T, T_mano)

    smpl_cam_data = None
    if smpl_path is not None:
        smpl_data = np.load(smpl_path, allow_pickle=True).item()
        smpl_cam_data = smpl_data[smpl_camera]
        T = min(T, smpl_cam_data['joints'].shape[0])

    pet_arr = pet_raw[:T].copy()

    # Match PetManoDataset / PetMotion preprocessing exactly: joints other
    # than joint 7 are local, while joint 7 keeps its global translation.
    transl = pet_arr[:, 7, :3].copy()
    pet_arr[:, :, :3] -= transl[:, None, :]
    pet_arr[:, 7, :3] = transl
    pet_flat = pet_arr[:, :, :3].reshape(T, -1)

    # Compute wrist midpoint and pet_rel
    if mano_data is not None:
        left_transl = mano_data['left']['transl'].astype(
            np.float32).reshape(-1, 3)[:T]
        right_transl = mano_data['right']['transl'].astype(
            np.float32).reshape(-1, 3)[:T]
        wrist_mid = (left_transl + right_transl) / 2.0
    else:
        # SMPL-only: use SMPL wrist joints (20=left_wrist, 21=right_wrist)
        smpl_joints = smpl_cam_data['joints'].astype(np.float32)[:T]
        wrist_mid = (smpl_joints[:, 20, :3] + smpl_joints[:, 21, :3]) / 2.0
    pet_rel = transl - wrist_mid

    # Normalize pet
    pm, ps = pet_stats['mean'], pet_stats['std']
    pet_normed = (pet_flat - pm) / ps

    prm = pet_stats['pet_rel_mean']
    prs = pet_stats['pet_rel_std']
    pet_rel_normed = (pet_rel - prm) / prs

    # Mano processing + normalize
    mano_normed = {}
    if mano_data is not None and mano_stats is not None:
        for hand, key_prefix in [('left', 'left'), ('right', 'right')]:
            joints = mano_data[hand]['joints'].astype(np.float32)[:T]
            if joints.ndim == 3 and joints.shape[1] != 21:
                joints = joints.squeeze(1)
            pose = mano_data[hand]['pose'].astype(np.float32).reshape(
                mano_data[hand]['joints'].shape[0], 16, 3, 3)[:T]
            wrist = joints[:, 0:1, :]
            rel_pos = (joints - wrist).reshape(T, -1)
            rot_flat = pose.reshape(T, -1)

            mm_p, ms_p = mano_stats[f'{key_prefix}_pos']
            mm_r, ms_r = mano_stats[f'{key_prefix}_rot']
            mano_normed[f'{key_prefix}_pos'] = (rel_pos - mm_p) / ms_p
            mano_normed[f'{key_prefix}_rot'] = (rot_flat - mm_r) / ms_r

    # SMPL processing + normalize
    smpl_normed = {}
    if smpl_cam_data is not None and smpl_stats is not None:
        joints = smpl_cam_data['joints'].astype(np.float32)[:T]
        body_pose = smpl_cam_data['body_pose'].astype(np.float32)[:T]
        global_orient = smpl_cam_data['global_orient'].astype(np.float32)[:T]
        root = joints[:, 0:1, :]
        smpl_rel_pos = (joints - root).reshape(T, -1)
        smpl_rot = np.concatenate([global_orient, body_pose], axis=-1)
        sm_p, ss_p = smpl_stats['pos']
        sm_r, ss_r = smpl_stats['rot']
        smpl_normed['pos'] = (smpl_rel_pos - sm_p) / ss_p
        smpl_normed['rot'] = (smpl_rot - sm_r) / ss_r

    # Audio loading
    audio_data = None
    if audio_path is not None and os.path.isfile(audio_path):
        audio_data = np.load(audio_path).astype(np.float32)

    # Chunk into non-overlapping windows
    n_chunks = T // seq_len
    chunks = {
        'pet': [], 'pet_rel': [],
    }
    if mano_normed:
        chunks.update({'left_pos': [], 'left_rot': [],
                       'right_pos': [], 'right_rot': []})
    if smpl_normed:
        chunks['smpl_pos'] = []
        chunks['smpl_rot'] = []
    if audio_data is not None:
        chunks['audio'] = []

    audio_ratio = audio_fps / motion_fps  # 75/30 = 2.5

    for i in range(n_chunks):
        s, e = i * seq_len, (i + 1) * seq_len
        chunks['pet'].append(pet_normed[s:e])
        chunks['pet_rel'].append(pet_rel_normed[s:e])
        if mano_normed:
            chunks['left_pos'].append(mano_normed['left_pos'][s:e])
            chunks['left_rot'].append(mano_normed['left_rot'][s:e])
            chunks['right_pos'].append(mano_normed['right_pos'][s:e])
            chunks['right_rot'].append(mano_normed['right_rot'][s:e])
        if smpl_normed:
            chunks['smpl_pos'].append(smpl_normed['pos'][s:e])
            chunks['smpl_rot'].append(smpl_normed['rot'][s:e])
        if audio_data is not None:
            a_start = int(s * audio_ratio)
            a_len = int(seq_len * audio_ratio)
            a_end = a_start + a_len
            if a_end <= len(audio_data):
                audio_chunk = audio_data[a_start:a_end]
            else:
                audio_chunk = np.zeros((a_len, audio_data.shape[-1]),
                                       dtype=np.float32)
                valid = min(len(audio_data) - a_start, a_len)
                if valid > 0:
                    audio_chunk[:valid] = audio_data[a_start:a_start + valid]
            chunks['audio'].append(audio_chunk)

    return {
        'chunks': chunks,
        'n_chunks': n_chunks,
        'pet_flat_full': pet_flat,
        'pm': pm, 'ps': ps,
        'T_aligned': T,
    }


def load_classifier(ckpt_path, device):
    """Load pretrained PetMotionClassifier."""
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = ckpt.get('config', None)
    if cfg is not None:
        cfg = EasyDict(cfg)
    if cfg is not None and hasattr(cfg, 'structure'):
        hps = EasyDict(cfg.structure)
    else:
        hps = EasyDict(input_dim=60, feat_dim=512, width=512, depth=4,
                        downs_t=[2], strides_t=[2])
    # num_classes doesn't matter for feature extraction
    if not hasattr(hps, 'num_classes'):
        hps.num_classes = 12
    model = PetMotionClassifier(hps).to(device)
    state = ckpt['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def extract_features_from_motions(classifier, motions_list, device):
    """Extract features from a list of (T, 60) numpy arrays.

    Splits each motion into 300-frame chunks, extracts per-chunk features.
    Returns (N_chunks, feat_dim) numpy array.
    """
    chunk_len = 300
    all_feats = []
    with torch.no_grad():
        for mot in motions_list:
            T = mot.shape[0]
            n = T // chunk_len
            for i in range(n):
                chunk = mot[i * chunk_len:(i + 1) * chunk_len]
                x = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
                feat = classifier.extract_features(x)  # (1, D)
                all_feats.append(feat.cpu().numpy())
    if len(all_feats) == 0:
        return np.zeros((0, 512))
    return np.concatenate(all_feats, axis=0)


def load_rprecision_encoder(ckpt_path, device):
    """Load trained RPrecisionEncoder from checkpoint."""
    # R-Precision is optional and its encoder still depends on legacy model
    # modules.  Keep the import lazy so FID-only Wan evaluation does not load
    # that legacy stack.
    from models.rprecision_encoder import RPrecisionEncoder

    ckpt = load_checkpoint(ckpt_path, device)
    cfg = EasyDict(ckpt['config'])
    m = cfg.model
    model = RPrecisionEncoder(
        dim_a=m.dim_a, dim_b=m.dim_b,
        emb_dim=m.get('emb_dim', 256),
        width=m.get('width', 512),
        depth=m.get('depth', 4),
        downs_t=m.get('downs_t', [2]),
        strides_t=m.get('strides_t', [2]),
        m_conv=m.get('m_conv', 1.0),
        dilation_growth_rate=m.get('dilation_growth_rate', 3),
        dropout=m.get('dropout', 0.0),
    ).to(device)
    state = ckpt['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg.mode


def compute_rprecision_from_embeddings(z_a, z_b, pool_size=32, num_trials=20):
    """Compute R-Precision top-1/top-3 from pre-computed embeddings.

    Args:
        z_a: (N, D) L2-normalized embeddings (pet)
        z_b: (N, D) L2-normalized embeddings (human)
        pool_size: retrieval pool size (1 GT + pool_size-1 negatives)
        num_trials: number of random trials to average
    Returns:
        top1, top3: float
    """
    N = z_a.shape[0]
    if N < pool_size:
        pool_size = N

    top1_total = 0
    top3_total = 0
    count = 0

    for _ in range(num_trials):
        perm = torch.randperm(N, device=z_a.device)
        for start in range(0, N - pool_size + 1, pool_size):
            idx = perm[start:start + pool_size]
            za_pool = z_a[idx]
            zb_pool = z_b[idx]
            sim = za_pool @ zb_pool.T
            ranks = (sim >= sim.diag().unsqueeze(1)).sum(dim=1)
            top1_total += (ranks == 1).sum().item()
            top3_total += (ranks <= 3).sum().item()
            count += pool_size

    top1 = top1_total / max(count, 1)
    top3 = top3_total / max(count, 1)
    return top1, top3


def parse_args():
    p = argparse.ArgumentParser(description='Test generation + FID')
    p.add_argument('--config', type=str, default='configs/pet_gpt.yaml',
                   help='GPT config yaml')
    p.add_argument('--gpt_ckpt', type=str, default=None,
                   help='GPT checkpoint (default: experiments/<expname>/ckpt/best_eval.pt)')
    p.add_argument('--classifier_ckpt', type=str,
                   default='experiments/pet_motion_classifier/ckpt/best_eval.pt')
    p.add_argument('--out_dir', type=str, default='outputs/test_fid')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--save_npy', action='store_true',
                   help='Save per-file generated .npy')
    p.add_argument('--rprecision_ckpt', type=str, default=None,
                   help='Trained RPrecisionEncoder checkpoint for R-Precision eval')
    p.add_argument('--split', type=str, default='val',
                   choices=['train', 'val', 'test'],
                   help='Data split to evaluate on (test is an alias for val)')
    p.add_argument('--split_manifest', type=str, default=None,
                   help='Optional YAML with explicit train_files/val_files')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    seed = int(getattr(config, 'seed', 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_smpl = getattr(config, 'use_smpl', False)
    use_mano = getattr(config, 'use_mano', True)
    use_audio = getattr(config.model, 'use_audio', False)
    d = config.data
    pet_root = d.pet_root
    mano_root = getattr(d, 'mano_root', None) if use_mano else None
    smpl_root = getattr(d, 'smpl_root', None) if use_smpl else None
    audio_root = getattr(d, 'audio_root', None) if use_audio else None
    seq_len = d.seq_len

    gpt_ckpt = args.gpt_ckpt or os.path.join(
        'experiments', config.expname, 'ckpt', 'best_eval.pt')

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load norm stats ──
    pet_stats, mano_stats, smpl_stats = load_norm_stats(
        pet_root, mano_root, smpl_root)

    # ── Get files for evaluation ──
    if args.split_manifest:
        with open(args.split_manifest) as manifest_file:
            manifest = yaml.safe_load(manifest_file)
        split_key = f'{args.split}_files'
        if split_key not in manifest:
            raise KeyError(
                f'{args.split_manifest} does not contain {split_key}')
        test_files = list(manifest[split_key])
        required_roots = [pet_root]
        required_roots += [root for root in (mano_root, smpl_root, audio_root)
                           if root is not None]
        missing = [
            f'{root}/{filename}'
            for filename in test_files
            for root in required_roots
            if not os.path.isfile(os.path.join(root, filename))
        ]
        if missing:
            preview = '\n'.join(missing[:10])
            raise FileNotFoundError(
                f'{len(missing)} manifest inputs are missing:\n{preview}')
    else:
        test_files = get_test_files(
            pet_root, mano_root, smpl_root, audio_root,
            test_ratio=d.test_ratio, split_seed=d.split_seed,
            split=args.split)
    print(f'{args.split} files: {len(test_files)}')

    # PetMotion and PetManoDataset map dog IDs by lexically sorting every dog
    # present in the data.  Decoding all samples as dog 0 corrupts the
    # identity-conditioned VQ output for dog02+.
    all_dog_ids = sorted({
        match.group(1)
        for fn in os.listdir(pet_root)
        if fn.endswith('.npy')
        for match in [re.search(r'(dog\d+)', fn)]
        if match is not None
    })
    dog_id_to_idx = {dog_id: idx for idx, dog_id in enumerate(all_dog_ids)}
    print(f'Dog ID mapping: {dog_id_to_idx}')

    # ── Load VQ-VAEs ──
    print('Loading PetVQVAE...')
    pet_vqvae = load_pretrained_vqvae(config.pet_vqvae_ckpt, device)
    for p in pet_vqvae.parameters():
        p.requires_grad = False

    print('Loading PetRelVQVAE...')
    pet_rel_vqvae = load_pretrained_vqvae(
        config.pet_rel_vqvae_ckpt, device)
    for p in pet_rel_vqvae.parameters():
        p.requires_grad = False

    mano_vqvae = None
    if use_mano:
        print('Loading ManoVQVAE...')
        mano_vqvae = load_pretrained_mano_vqvae(config.mano_vqvae_ckpt, device)
        for p in mano_vqvae.parameters():
            p.requires_grad = False

    smpl_vqvae = None
    if use_smpl:
        print('Loading SmplVQVAE...')
        smpl_vqvae = load_pretrained_vqvae(
            config.smpl_vqvae_ckpt, device, model_cls=WanSmplVQVAE)
        for p in smpl_vqvae.parameters():
            p.requires_grad = False

    # ── Load GPT ──
    print(f'Loading PetGPT from {gpt_ckpt}...')
    gpt_state = load_checkpoint(gpt_ckpt, device)
    gpt_saved_config = EasyDict(gpt_state['config'])
    gpt_model_cfg = EasyDict(gpt_saved_config.model)
    gpt = PetGPT(gpt_model_cfg).to(device)
    state = gpt_state['model']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    gpt.load_state_dict(state)
    gpt.eval()

    pet_code_len = gpt_model_cfg.pet_len
    print(f'GPT pet_len={pet_code_len}, use_mano={use_mano}, '
          f'use_smpl={getattr(gpt_model_cfg, "use_smpl", False)}, '
          f'use_audio={use_audio}')

    # ── Load classifier for FID ──
    print(f'Loading classifier from {args.classifier_ckpt}...')
    classifier = load_classifier(args.classifier_ckpt, device)

    # ── Denorm helpers ──
    pm_t = torch.from_numpy(pet_stats['mean']).to(device)
    ps_t = torch.from_numpy(pet_stats['std']).to(device)

    # ── Load R-Precision encoder (optional) ──
    rprecision_model = None
    rprecision_mode = None
    if args.rprecision_ckpt:
        print(f'Loading RPrecisionEncoder from {args.rprecision_ckpt}...')
        rprecision_model, rprecision_mode = load_rprecision_encoder(
            args.rprecision_ckpt, device)
        for p_param in rprecision_model.parameters():
            p_param.requires_grad = False
        print(f'  mode={rprecision_mode}')

    # ── Generate on all test files ──
    gt_motions_all = []      # denormalized (T, 60) numpy
    gen_motions_all = []     # denormalized (T, 60) numpy
    vq_recon_motions_all = []  # VQ reconstruction baseline

    # For R-Precision: collect normalized chunks
    rp_gt_pet_chunks = []     # normalized GT pet (seq_len, 60)
    rp_gen_pet_chunks = []    # normalized generated pet (seq_len, 60)
    rp_human_chunks = []      # normalized human motion (seq_len, dim_b)

    skipped = 0

    for fn in tqdm(test_files, desc='Generating'):
        data = preprocess_sequence(
            os.path.join(pet_root, fn),
            os.path.join(mano_root, fn) if mano_root else None,
            pet_stats, mano_stats, seq_len=seq_len,
            smpl_path=os.path.join(smpl_root, fn) if smpl_root else None,
            smpl_stats=smpl_stats,
            audio_path=os.path.join(audio_root, fn) if audio_root else None,
        )
        n_chunks = data['n_chunks']
        if n_chunks == 0:
            skipped += 1
            continue
        chunks = data['chunks']

        # Determine dog_id
        m_dog = re.search(r'(dog\d+)', fn)
        if m_dog is None or m_dog.group(1) not in dog_id_to_idx:
            raise ValueError(f'Cannot determine dog ID for {fn}')
        dog_id_t = torch.tensor(
            [dog_id_to_idx[m_dog.group(1)]], device=device).long()

        gen_motions_file = []
        vq_recon_file = []

        with torch.no_grad():
            for i in range(n_chunks):
                pet_t = torch.from_numpy(
                    chunks['pet'][i]).unsqueeze(0).float().to(device)
                pet_rel_t = torch.from_numpy(
                    chunks['pet_rel'][i]).unsqueeze(0).float().to(device)

                # Encode MANO conditioning
                left_codes = None
                right_codes = None
                if use_mano and mano_vqvae is not None and 'left_pos' in chunks:
                    lp_t = torch.from_numpy(
                        chunks['left_pos'][i]).unsqueeze(0).float().to(device)
                    lr_t = torch.from_numpy(
                        chunks['left_rot'][i]).unsqueeze(0).float().to(device)
                    rp_t = torch.from_numpy(
                        chunks['right_pos'][i]).unsqueeze(0).float().to(device)
                    rr_t = torch.from_numpy(
                        chunks['right_rot'][i]).unsqueeze(0).float().to(device)
                    left_codes = mano_vqvae.vqvae_left.encode(
                        lp_t, lr_t)[0].long()
                    right_codes = mano_vqvae.vqvae_right.encode(
                        rp_t, rr_t)[0].long()

                smpl_codes = None
                if use_smpl and smpl_vqvae is not None and 'smpl_pos' in chunks:
                    sp_t = torch.from_numpy(
                        chunks['smpl_pos'][i]).unsqueeze(0).float().to(device)
                    sr_t = torch.from_numpy(
                        chunks['smpl_rot'][i]).unsqueeze(0).float().to(device)
                    smpl_codes = smpl_vqvae.encode(sp_t, sr_t)[0].long()

                # Audio features
                audio_feats = None
                if use_audio and 'audio' in chunks:
                    audio_feats = torch.from_numpy(
                        chunks['audio'][i]).unsqueeze(0).float().to(device)

                # Encode GT pet codes (for VQ recon baseline)
                pet_codes_gt = pet_vqvae.encode(pet_t)[0].long()

                # Generate
                _, gen_pet_codes = gpt.generate(
                    left_codes, right_codes,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    smpl_codes=smpl_codes,
                    audio_features=audio_feats,
                )

                # Decode
                gen_mot = pet_vqvae.decode(
                    [gen_pet_codes], dog_id=dog_id_t)  # (1, T, 60)
                vq_mot = pet_vqvae.decode(
                    [pet_codes_gt], dog_id=dog_id_t)

                # Denormalize
                gen_denorm = (gen_mot * ps_t + pm_t).cpu().numpy()[0]
                vq_denorm = (vq_mot * ps_t + pm_t).cpu().numpy()[0]

                gen_motions_file.append(gen_denorm)
                vq_recon_file.append(vq_denorm)

                # Collect for R-Precision (normalized chunks)
                if rprecision_model is not None:
                    rp_gt_pet_chunks.append(chunks['pet'][i])      # (seq_len, 60)
                    rp_gen_pet_chunks.append(gen_mot[0].cpu().numpy())  # (seq_len, 60) normalized

                    if rprecision_mode == 'pet_mano' and 'left_pos' in chunks:
                        human_chunk = np.concatenate([
                            chunks['left_pos'][i],
                            chunks['left_rot'][i],
                            chunks['right_pos'][i],
                            chunks['right_rot'][i],
                        ], axis=-1)  # (seq_len, 414)
                    elif 'smpl_pos' in chunks:  # pet_smpl
                        human_chunk = np.concatenate([
                            chunks['smpl_pos'][i],
                            chunks['smpl_rot'][i],
                        ], axis=-1)  # (seq_len, 207)
                    else:
                        human_chunk = None
                    if human_chunk is None:
                        continue
                    rp_human_chunks.append(human_chunk)

        # GT raw (denormalized)
        T_out = n_chunks * seq_len
        gt_flat = data['pet_flat_full'][:T_out]

        gt_motions_all.append(gt_flat)
        gen_motions_all.append(np.concatenate(gen_motions_file, axis=0))
        vq_recon_motions_all.append(np.concatenate(vq_recon_file, axis=0))

        # Optionally save per-file
        if args.save_npy:
            fname_base = fn.replace('.npy', '')
            np.save(os.path.join(args.out_dir, f'{fname_base}_gt.npy'), gt_flat)
            np.save(os.path.join(args.out_dir, f'{fname_base}_gen.npy'),
                    gen_motions_all[-1])

    print(f'\nGenerated motions from {len(test_files) - skipped} files '
          f'(skipped {skipped} files with < {seq_len} frames)')

    # ── Compute FID using the released historical evaluation convention ──
    print('\nExtracting GT/generated/VQ features...')
    feats_gt = extract_features_from_motions(
        classifier, gt_motions_all, device)
    feats_gen = extract_features_from_motions(
        classifier, gen_motions_all, device)
    feats_vq = extract_features_from_motions(
        classifier, vq_recon_motions_all, device)
    print(f'  Raw feature shapes: GT={feats_gt.shape}, '
          f'Gen={feats_gen.shape}, VQ={feats_vq.shape}')
    fid_gen = compute_pet_fid(feats_gen, feats_gt)
    fid_vq = compute_pet_fid(feats_vq, feats_gt)

    # ── Compute Diversity ──
    def compute_diversity(feats, num_pairs=200, seed=42):
        """Average L2 distance between randomly sampled pairs of features."""
        N = feats.shape[0]
        if N < 2:
            return 0.0
        rng = np.random.RandomState(seed)
        num_pairs = min(num_pairs, N * (N - 1) // 2)
        idx_a = rng.randint(0, N, size=num_pairs)
        idx_b = rng.randint(0, N, size=num_pairs)
        # Avoid same index
        mask = idx_a == idx_b
        idx_b[mask] = (idx_b[mask] + 1) % N
        dists = np.linalg.norm(feats[idx_a] - feats[idx_b], axis=-1)
        return float(dists.mean())

    div_gen = compute_diversity(feats_gen)
    div_gt = compute_diversity(feats_gt)

    print(f'\n{"="*50}')
    print(f'FID (Generated vs GT):     {fid_gen:.4f}')
    print(f'FID (VQ Recon vs GT):      {fid_vq:.4f}')
    print(f'Diversity (Generated):   {div_gen:.4f}')
    print(f'Diversity (GT):          {div_gt:.4f}')
    print(f'{"="*50}')

    # ── Compute R-Precision ──
    rprec_results = {}
    if rprecision_model is not None and len(rp_gt_pet_chunks) > 0:
        print(f'\nComputing R-Precision ({rprecision_mode}, '
              f'{len(rp_gt_pet_chunks)} chunks)...')

        encode_batch_size = 256
        gt_za_list, gen_za_list, zb_list = [], [], []
        with torch.no_grad():
            for start in range(0, len(rp_gt_pet_chunks), encode_batch_size):
                end = min(start + encode_batch_size, len(rp_gt_pet_chunks))
                gt_pet_batch = torch.from_numpy(
                    np.stack(rp_gt_pet_chunks[start:end])).float().to(device)
                gen_pet_batch = torch.from_numpy(
                    np.stack(rp_gen_pet_chunks[start:end])).float().to(device)
                human_batch = torch.from_numpy(
                    np.stack(rp_human_chunks[start:end])).float().to(device)

                gt_za_list.append(rprecision_model.encode_a(gt_pet_batch))
                gen_za_list.append(rprecision_model.encode_a(gen_pet_batch))
                zb_list.append(rprecision_model.encode_b(human_batch))

        gt_za = torch.cat(gt_za_list, dim=0)
        gen_za = torch.cat(gen_za_list, dim=0)
        zb = torch.cat(zb_list, dim=0)

        # GT pet vs human (upper bound)
        gt_top1, gt_top3 = compute_rprecision_from_embeddings(gt_za, zb)
        # Generated pet vs human (our metric)
        gen_top1, gen_top3 = compute_rprecision_from_embeddings(gen_za, zb)

        print(f'R-Precision (GT pet vs human):   top1={gt_top1:.4f}  top3={gt_top3:.4f}')
        print(f'R-Precision (Gen pet vs human):  top1={gen_top1:.4f}  top3={gen_top3:.4f}')

        rprec_results = {
            'rprecision_gt_top1': gt_top1,
            'rprecision_gt_top3': gt_top3,
            'rprecision_gen_top1': gen_top1,
            'rprecision_gen_top3': gen_top3,
            'rprecision_mode': rprecision_mode,
            'rprecision_ckpt': args.rprecision_ckpt,
        }

    # Save results
    results = {
        'fid_gen_vs_gt': fid_gen,
        'fid_vq_recon_vs_gt': fid_vq,
        'diversity_gen': div_gen,
        'diversity_gt': div_gt,
        'n_test_files': len(test_files) - skipped,
        'split_manifest': args.split_manifest,
        'config': args.config,
        'gpt_ckpt': gpt_ckpt,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'use_smpl': use_smpl,
        'use_mano': use_mano,
        **rprec_results,
    }
    results_path = os.path.join(args.out_dir, 'fid_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
