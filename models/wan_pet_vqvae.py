"""
WanPetVQVAE: VQ-VAE for pet 3D keypoints using Wan2.2-style bidirectional
encoder/decoder with EMA VQ bottleneck.
Input: (B, T, n_kp*3)  Output: (B, T, n_kp*3)
"""
import numpy as np
import torch
import torch.nn as nn

from .wan_encdec import WanEncoder1d, WanDecoder1d
from .bottleneck import Bottleneck, NoBottleneck


def _loss_fn_l1(x_target, x_pred):
    return torch.mean(torch.abs(x_pred - x_target))


# Foot (paw) joint indices in the 20-joint animal skeleton
# 16, 17 = back paws; 18, 19 = front paws
FOOT_JOINT_IDS = [16, 17, 18, 19]


def _foot_sliding_loss(x_out, x_target, foot_ids=FOOT_JOINT_IDS,
                       height_axis=2, height_thresh=0.02):
    """Penalize horizontal velocity of foot joints when they are near the ground.

    Ground plane is estimated per-sample as the 5th percentile of GT foot
    heights. Contact is detected when a foot is within ``height_thresh``
    above this estimated ground.

    Args:
        x_out:    (B, T, 60) reconstructed motion (20 joints * 3)
        x_target: (B, T, 60) ground-truth motion
        foot_ids: list of joint indices considered as feet
        height_axis: which of the 3 xyz axes is height (0=x,1=y,2=z)
        height_thresh: height above estimated ground below which foot is grounded
    Returns:
        scalar loss
    """
    B, T, _ = x_out.shape
    pred = x_out.reshape(B, T, 20, 3)
    gt = x_target.reshape(B, T, 20, 3)

    # Extract foot joints: (B, T, n_feet, 3)
    pred_feet = pred[:, :, foot_ids, :]
    gt_feet = gt[:, :, foot_ids, :]

    # GT foot heights: (B, T, n_feet)
    gt_height = gt_feet[:, :, :, height_axis]

    # Estimate ground per sample as 5th percentile of foot heights
    # (B, T*n_feet) -> (B,)
    gt_height_flat = gt_height.reshape(B, -1)
    ground = torch.quantile(gt_height_flat, 0.05, dim=1)  # (B,)
    # Height above ground: (B, T, n_feet)
    rel_height = gt_height - ground[:, None, None]

    # Soft contact mask: sigmoid((thresh - rel_h) / temperature)
    contact = torch.sigmoid((height_thresh - rel_height) / 0.005)

    # Horizontal velocity of reconstructed feet: (B, T-1, n_feet, 3)
    pred_vel = pred_feet[:, 1:] - pred_feet[:, :-1]

    # Zero out height axis — only penalize horizontal sliding
    horiz_mask = torch.ones(3, device=x_out.device)
    horiz_mask[height_axis] = 0.0
    pred_vel_horiz = pred_vel * horiz_mask

    # Weighted L1
    contact_t = contact[:, 1:]  # (B, T-1, n_feet)
    sliding = torch.abs(pred_vel_horiz) * contact_t.unsqueeze(-1)

    return sliding.mean()


class WanPetVQVAE(nn.Module):
    """VQ-VAE with Wan-style backbone for pet motion.

    Config (hps) fields:
        input_dim:       total input channels (e.g. 60 = 20*3)
        latent_dim:      encoder output / VQ embedding width (default 128)
        l_bins:          codebook size (default 512)
        l_mu:            EMA decay (default 0.99)
        commit:          commitment loss weight
        vel:             velocity loss weight
        acc:             acceleration loss weight
        bone_weight:     bone length loss weight
        num_bones:       number of bones for bone loss
        foot_sliding_weight: foot sliding loss weight (default 0.0)
        foot_height_thresh:  height threshold for contact detection (default 0.05)
        wan_dim:         base channel width (default 128)
        wan_dim_mult:    channel multipliers (default [1,2,4,4])
        wan_num_res_blocks: res blocks per stage (default 2)
        wan_temporal_downsample: downsample flags (default [True,True,False])
        wan_dropout:     dropout (default 0.0)
        sample_length:   expected input sequence length
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.input_dim = getattr(hps, 'input_dim', 60)

        # Wan backbone params
        wan_dim = getattr(hps, 'wan_dim', 128)
        wan_dim_mult = tuple(getattr(hps, 'wan_dim_mult', [1, 2, 4, 4]))
        wan_num_res = getattr(hps, 'wan_num_res_blocks', 2)
        wan_t_down = list(getattr(hps, 'wan_temporal_downsample', [True, True, False]))
        wan_dropout = getattr(hps, 'wan_dropout', 0.0)

        # VQ params
        latent_dim = getattr(hps, 'latent_dim', 128)
        l_bins = getattr(hps, 'l_bins', 512)
        mu = getattr(hps, 'l_mu', 0.99)
        self.commit = getattr(hps, 'commit', 0.02)
        self.vel = getattr(hps, 'vel', 1.0)
        self.acc = getattr(hps, 'acc', 1.0)
        self.foot_sliding_weight = getattr(hps, 'foot_sliding_weight', 0.0)
        self.foot_height_thresh = getattr(hps, 'foot_height_thresh', 0.05)
        self.sample_length = getattr(hps, 'sample_length', 300)

        # Compute temporal downsample factor
        self.temporal_downsample_factor = 2 ** sum(wan_t_down)

        # Encoder
        self.encoder = WanEncoder1d(
            input_dim=self.input_dim,
            dim=wan_dim,
            z_dim=latent_dim,
            dim_mult=wan_dim_mult,
            num_res_blocks=wan_num_res,
            temporal_downsample=wan_t_down,
            dropout=wan_dropout,
        )

        # VQ Bottleneck (single level)
        use_bottleneck = getattr(hps, 'use_bottleneck', True)
        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, latent_dim, mu, levels=1)
        else:
            self.bottleneck = NoBottleneck(levels=1)

        # Decoder
        wan_t_up = wan_t_down[::-1]
        self.decoder = WanDecoder1d(
            output_dim=self.input_dim,
            dim=wan_dim,
            z_dim=latent_dim,
            dim_mult=wan_dim_mult,
            num_res_blocks=wan_num_res,
            temporal_upsample=wan_t_up,
            dropout=wan_dropout,
        )

        self.l_bins = l_bins
        self.latent_dim = latent_dim

        # Dog identity embedding
        self.num_dogs = getattr(hps, 'num_dogs', 0)
        if self.num_dogs > 0:
            self.dog_emb = nn.Embedding(self.num_dogs, latent_dim)

        # Bone length decoder head
        self.num_bones = getattr(hps, 'num_bones', 20)
        self.bone_weight = getattr(hps, 'bone_weight', 0.0)
        if self.bone_weight > 0:
            self.bone_decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, self.num_bones),
            )

    def forward(self, x, dog_id=None, bone_lengths=None):
        """x: (B, T, input_dim). Returns (x_out, loss, metrics)."""
        metrics = {}
        x_target = x.float()
        B, T, C = x_target.shape

        # Encode: NTC -> NCT
        x_in = x_target.permute(0, 2, 1)  # (B, C, T)
        z_e = self.encoder(x_in)           # (B, latent_dim, T_down)

        # Quantize
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck([z_e])

        # Bone length prediction
        bone_loss = torch.zeros((), device=x.device)
        if self.bone_weight > 0 and bone_lengths is not None:
            z_pooled = xs_quantised[0].mean(dim=-1)  # (B, latent_dim)
            bone_pred = self.bone_decoder(z_pooled)
            bone_loss = _loss_fn_l1(bone_lengths, bone_pred)
            metrics['bone_loss'] = bone_loss.detach()

        # Add dog identity embedding
        z_q = xs_quantised[0]  # (B, latent_dim, T_down)
        if self.num_dogs > 0 and dog_id is not None:
            d_emb = self.dog_emb(dog_id).unsqueeze(-1)  # (B, latent_dim, 1)
            z_q = z_q + d_emb

        # Decode: NCT -> NTC
        x_dec = self.decoder(z_q)          # (B, C, T_recon)
        x_out = x_dec.permute(0, 2, 1)    # (B, T_recon, C)

        # Trim to original length (in case of padding from downsample/upsample)
        if x_out.shape[1] > T:
            x_out = x_out[:, :T, :]
        elif x_out.shape[1] < T:
            x_out = F.pad(x_out, (0, 0, 0, T - x_out.shape[1]))

        # Losses
        recons_loss = _loss_fn_l1(x_target, x_out)
        velocity_loss = _loss_fn_l1(
            x_out[:, 1:] - x_out[:, :-1],
            x_target[:, 1:] - x_target[:, :-1],
        )
        acceleration_loss = _loss_fn_l1(
            x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1],
            x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1],
        )
        commit_loss = sum(commit_losses)

        # Foot sliding loss
        foot_sliding_loss = torch.zeros((), device=x.device)
        if self.foot_sliding_weight > 0:
            foot_sliding_loss = _foot_sliding_loss(
                x_out, x_target, height_thresh=self.foot_height_thresh)
            metrics['foot_sliding_loss'] = foot_sliding_loss.detach()

        loss = (
            recons_loss
            + self.commit * commit_loss
            + self.vel * velocity_loss
            + self.acc * acceleration_loss
            + self.bone_weight * bone_loss
            + self.foot_sliding_weight * foot_sliding_loss
        )

        metrics.update(dict(
            recons_loss=recons_loss.detach(),
            commit_loss=commit_loss.detach(),
            velocity_loss=velocity_loss.detach(),
            acceleration_loss=acceleration_loss.detach(),
        ))
        if quantiser_metrics:
            for k, v in quantiser_metrics[0].items():
                metrics[f'vq_{k}'] = v.detach() if hasattr(v, 'detach') else v

        return x_out, loss, metrics

    def encode(self, x):
        """x: (B, T, input_dim). Returns list of code indices [(B, T_down)]."""
        x_in = x.permute(0, 2, 1).float()
        z_e = self.encoder(x_in)
        zs = self.bottleneck.encode([z_e])
        return zs

    def decode(self, zs, dog_id=None):
        """zs: list of code indices. Returns (B, T, input_dim)."""
        xs_quantised = self.bottleneck.decode(zs, start_level=0, end_level=1)
        z_q = xs_quantised[0]
        if self.num_dogs > 0 and dog_id is not None:
            d_emb = self.dog_emb(dog_id).unsqueeze(-1)
            z_q = z_q + d_emb
        x_out = self.decoder(z_q)
        return x_out.permute(0, 2, 1)
