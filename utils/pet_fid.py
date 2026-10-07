"""FID computation utility for pet motion using a trained classifier's features."""

import numpy as np
import torch
from tqdm import tqdm


def extract_pet_features(model, dataloader, device):
    """Extract penultimate-layer features from a trained PetMotionClassifier.

    Args:
        model: PetMotionClassifier (already on device, in eval mode).
        dataloader: yields dicts with 'motion' key of shape (B, T, 60).
        device: torch device.
    Returns:
        feats: (N, feat_dim) numpy array.
    """
    model.eval()
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            motion = batch['motion'].float().to(device)
            feats = model.extract_features(motion)  # (B, feat_dim)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_pet_fid(feats_gen, feats_gt):
    """Compute FID between two sets of features.

    Args:
        feats_gen: (N1, D) numpy array of generated features.
        feats_gt: (N2, D) numpy array of ground-truth features.
    Returns:
        fid: scalar FID value.
    """
    from scipy import linalg

    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    mu_gt = np.mean(feats_gt, axis=0)
    sigma_gt = np.cov(feats_gt, rowvar=False)

    diff = mu_gen - mu_gt
    covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_gt), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-5
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_gt) - 2 * np.trace(covmean)
    return float(fid)


if __name__ == '__main__':
    import argparse
    import yaml
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from easydict import EasyDict
    from datasets.pet_motion import PetMotion
    from models.pet_motion_classifier import PetMotionClassifier

    parser = argparse.ArgumentParser(description='Compute pet motion FID')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained classifier checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/pet_motion_classifier.yaml')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root')
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.safe_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = args.data_root or config.data.train.data_root

    # Load datasets
    trainset = PetMotion(
        data_root=data_root,
        seq_len=config.data.train.seq_len,
        split='train',
        stride=config.data.train.get('stride', config.data.train.seq_len),
        test_ratio=config.data.get('test_ratio', 0.2),
        split_seed=config.data.get('split_seed', 42),
    )
    valset = PetMotion(
        data_root=data_root,
        seq_len=config.data.test.seq_len,
        split='val',
        stride=config.data.test.get('stride', config.data.test.seq_len),
        norm_stats_path=trainset.norm_stats_path,
        test_ratio=config.data.get('test_ratio', 0.2),
        split_seed=config.data.get('split_seed', 42),
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=128, shuffle=False, drop_last=False)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', config)
    if not hasattr(ckpt_config, 'structure'):
        ckpt_config = config
    ckpt_config.structure.num_classes = trainset.num_dogs
    model = PetMotionClassifier(ckpt_config.structure).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Extract features
    print('Extracting train features...')
    feats_train = extract_pet_features(model, train_loader, device)
    print(f'  shape: {feats_train.shape}')

    print('Extracting val features...')
    feats_val = extract_pet_features(model, val_loader, device)
    print(f'  shape: {feats_val.shape}')

    # Compute FID
    fid = compute_pet_fid(feats_val, feats_train)
    print(f'FID (val vs train): {fid:.4f}')

    # Self-FID (sanity check: should be ~0)
    fid_self = compute_pet_fid(feats_train, feats_train)
    print(f'FID (train vs train, sanity): {fid_self:.4f}')
