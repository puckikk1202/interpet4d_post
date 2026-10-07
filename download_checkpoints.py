"""Download and verify the public PetGPT-SMPL checkpoints."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
REPO_ID = "ohicarip/interpet4d"
REPO_TYPE = "dataset"
REMOTE_DIR = "model_weights/petgpt_smpl/checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    metadata = json.loads((ROOT / "CHECKPOINT_METADATA.json").read_text())
    destination = ROOT / "checkpoints"
    destination.mkdir(exist_ok=True)

    for filename, details in metadata.items():
        print(f"Downloading {filename} ...")
        cached = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f"{REMOTE_DIR}/{filename}",
        ))
        target = destination / filename
        shutil.copyfile(cached, target)
        actual = sha256(target)
        expected = details["sha256"]
        if actual != expected:
            target.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: {actual} != {expected}")
        print(f"Verified {filename}: {actual}")

    print(f"All checkpoints are ready in {destination}")


if __name__ == "__main__":
    main()
