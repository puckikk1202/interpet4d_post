"""Offline integrity and import smoke test for the public release."""

from pathlib import Path
import json

import torch
import yaml
from easydict import EasyDict

from models.pet_gpt import PetGPT
from models.pet_motion_classifier import PetMotionClassifier
from models.wan_mano_vqvae import WanManoVQVAE
from models.wan_pet_vqvae import WanPetVQVAE
from models.wan_smpl_vqvae import WanSmplVQVAE


ROOT = Path(__file__).resolve().parent


def checkpoint(name):
    path = ROOT / "checkpoints" / name
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload["model"], dict)
    assert all(isinstance(value, torch.Tensor)
               for value in payload["model"].values())
    assert "/gs/" not in json.dumps(payload["config"])
    return payload


def main():
    with (ROOT / "configs/pet_gpt_smpl.yaml").open() as handle:
        release_config = EasyDict(yaml.safe_load(handle))

    gpt_payload = checkpoint("pet_gpt_smpl_best.pt")
    gpt = PetGPT(EasyDict(gpt_payload["config"]["model"]))
    gpt.load_state_dict(gpt_payload["model"])
    del gpt, gpt_payload

    for name, cls in (
        ("wan_pet_vqvae_global_best.pt", WanPetVQVAE),
        ("wan_pet_rel_vqvae_best.pt", WanPetVQVAE),
        ("wan_smpl_vqvae_best.pt", WanSmplVQVAE),
    ):
        payload = checkpoint(name)
        model = cls(EasyDict(payload["config"]["structure"]))
        model.load_state_dict(payload["model"])
        del model, payload

    mano_payload = checkpoint("wan_mano_vqvae_best.pt")
    mano_structure = EasyDict(mano_payload["config"]["structure"])
    mano = WanManoVQVAE(
        EasyDict(mano_structure.left), EasyDict(mano_structure.right))
    mano.load_state_dict(mano_payload["model"])
    del mano, mano_payload

    classifier_payload = checkpoint("pet_motion_classifier_best.pt")
    classifier = PetMotionClassifier(
        EasyDict(classifier_payload["config"]["structure"]))
    classifier.load_state_dict(classifier_payload["model"])
    del classifier, classifier_payload

    assert release_config.pet_vqvae_ckpt.startswith("checkpoints/")
    print("Smoke test passed: imports, safe checkpoint loading, and state dicts")


if __name__ == "__main__":
    main()
