# InterPet4D PetGPT-SMPL

This repository contains the model and evaluation code used for the public
PetGPT-SMPL result. The model conditions on MANO, SMPL, and MERT features and
generates 20-joint pet motion through the frozen Wan VQ stack.

- Dataset and model weights: https://huggingface.co/datasets/ohicarip/interpet4d
- Source code: https://github.com/puckikk1202/interpet4d_post

## Download checkpoints

Install the dependencies, then download and verify all six checkpoints:

```bash
python download_checkpoints.py
```

The script downloads from
`ohicarip/interpet4d/model_weights/petgpt_smpl/` and writes:

- `checkpoints/pet_gpt_smpl_best.pt`
- `checkpoints/wan_pet_vqvae_global_best.pt`
- `checkpoints/wan_pet_rel_vqvae_best.pt`
- `checkpoints/wan_mano_vqvae_best.pt`
- `checkpoints/wan_smpl_vqvae_best.pt`
- `checkpoints/pet_motion_classifier_best.pt`

The public checkpoints contain model weights, plain-dictionary architecture
metadata, and evaluation metadata only. Optimizer states and internal paths
have been removed. `download_checkpoints.py` verifies every file against
`CHECKPOINT_METADATA.json` before returning successfully.

## Environment

Python 3.10 and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version separately if the
default pip build is not suitable.

## Expected data layout

The dataset is not included. Arrange the aligned inputs as follows and update
the four paths in `configs/pet_gpt_smpl.yaml` if needed:

```text
data/
├── pet_npy/          # pet .npy files and norm_stats.json
├── mano_npy/         # MANO .npy files and mano_norm_stats.json
├── smpl_npy/         # SMPL .npy files and smpl_norm_stats.json
└── interpet_mert/    # aligned MERT features
```

Each modality uses the same sequence filename. Evaluation expects 30 fps
motion and 75 fps audio features.

## Train/test splits

The exact sequence lists used to train and evaluate every released model are:

| Model | Train | Test | Split files |
|---|---:|---:|---|
| PetGPT-SMPL | 146 | 36 | `train_files.txt`, `test_files.txt` |
| WanPetVQVAE | 187 | 40 | `pet_vqvae_train_files.txt`, `pet_vqvae_test_files.txt` |
| WanPetRelVQVAE | 146 | 36 | `train_files.txt`, `test_files.txt` |
| WanManoVQVAE | 146 | 36 | `train_files.txt`, `test_files.txt` |
| WanSmplVQVAE | 146 | 36 | `train_files.txt`, `test_files.txt` |

All files are under `splits/`. The 146/36 split is shared because the
PetGPT-SMPL, pet-relative, MANO, and SMPL models use the same 182 aligned
sequences. WanPetVQVAE additionally uses 45 pet-only sequences, giving its
187/40 split. The corresponding YAML manifests contain the same names for
machine-readable use. In both manifests, `test_files` and `val_files` are
identical aliases for the held-out evaluation subset.

For generation evaluation, pass `--split train`, `--split test`, or the
legacy `--split val` to select the desired PetGPT-SMPL list.

## Reproduce FID

Run from the root of this release directory:

```bash
python test_generation.py \
  --config configs/pet_gpt_smpl.yaml \
  --gpt_ckpt checkpoints/pet_gpt_smpl_best.pt \
  --classifier_ckpt checkpoints/pet_motion_classifier_best.pt \
  --split_manifest splits/petgpt_smpl.yaml \
  --split test \
  --out_dir outputs/petgpt_smpl \
  --temperature 1.0 \
  --top_k 50 \
  --save_npy
```

Reference result with seed 42:

| Metric | Value |
|---|---:|
| PetGPT FID | 5.0092 |
| VQ reconstruction FID | 0.6212 |

See `MODEL_CARD.md` for intended use and limitations.
