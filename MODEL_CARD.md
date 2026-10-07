# Model card: InterPet4D PetGPT-SMPL

## Model description

PetGPT-SMPL is an autoregressive prefix language model over frozen motion VQ
tokens. It uses left/right MANO motion, SMPL body motion, and aligned MERT
audio features as conditioning and predicts pet-relative-position tokens and
pet-motion tokens.

## Intended use

The model is intended for research on human-pet motion generation, benchmark
reproduction, ablation studies, and qualitative visualization. It is not
validated for safety-critical, veterinary, biometric, surveillance, or
real-time control applications.

## Training and evaluation data

The matched PetGPT-SMPL subset contains 182 sequences split deterministically
by dog identity into 146 training and 36 validation files with seed 42. The
release includes filenames only; it does not redistribute the underlying pet,
MANO, SMPL, audio, or participant data.

WanPetRelVQVAE, WanManoVQVAE, and WanSmplVQVAE use the same 146/36 aligned
split. WanPetVQVAE is trained on the full 227-sequence pet-motion set, split
into 187 training and 40 validation files. Exact train/test filenames for both
split families are included under `splits/`.

## Evaluation settings

- random seed: 42
- sampling temperature: 1.0
- top-k: 50
- classifier checkpoint accuracy: 0.9642857
- PetGPT FID: 5.00918468
- VQ reconstruction FID: 0.62121353

Compare results only when classifier weights, preprocessing, and split match.

## Limitations

- Evaluation is based on a small, identity-structured dataset.
- Autoregressive sampling is stochastic even with fixed hyperparameters; the
  released evaluator fixes all available random seeds for reproducibility.
- The model may generate implausible poses, temporal jitter, identity drift,
  or motion that is weakly aligned with the human conditioning signal.
- Performance outside the released dog identities, capture setup, skeleton
  definition, frame rates, or normalization statistics is unknown.
- The model has not been evaluated for demographic, participant, or animal
  representation bias.

## Privacy and redistribution

No raw participant data, audio, motion sequences, W&B logs, usernames, API
keys, or internal filesystem paths are included in the release package.
Confirm that the dataset consent and redistribution terms permit the intended
public use before publishing any example inputs or outputs.
