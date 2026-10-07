"""
WanManoVQVAE: left + right hand VQ-VAE using Wan-style bidirectional
encoder/decoder with EMA VQ bottleneck.

Each hand: 1 WanEncoder (pos+rot concat) → VQ → 2 WanDecoders (pos, rot)
"""
import numpy as np
import torch
import torch.nn as nn

from .wan_encdec import WanEncoder1d, WanDecoder1d
from .bottleneck import Bottleneck, NoBottleneck


def _loss_fn_l1(x_target, x_pred):
    return torch.mean(torch.abs(x_pred - x_target))


class WanManoHandVQVAE(nn.Module):
    """Single-hand VQ-VAE with Wan-style backbone.

    Config (hps) fields:
        pos_channel:    joint position dim (default 63 = 21*3)
        rot_channel:    rotation matrix dim (default 144 = 16*9)
        latent_dim:     encoder output / VQ embedding width
        l_bins:         codebook size
        l_mu:           EMA decay
        commit:         commitment loss weight
        vel:            velocity loss weight
        acc:            acceleration loss weight
        wan_dim:        base channel width
        wan_dim_mult:   channel multipliers
        wan_num_res_blocks: res blocks per stage
        wan_temporal_downsample: downsample flags
        wan_dropout:    dropout
        sample_length:  expected input sequence length
    """

    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        pos_channel = getattr(hps, 'pos_channel', 63)
        rot_channel = getattr(hps, 'rot_channel', 144)
        self.pos_channel = pos_channel
        self.rot_channel = rot_channel
        enc_input = pos_channel + rot_channel

        # Wan backbone params
        wan_dim = getattr(hps, 'wan_dim', 128)
        wan_dim_mult = tuple(getattr(hps, 'wan_dim_mult', [1, 2, 4, 4]))
        wan_num_res = getattr(hps, 'wan_num_res_blocks', 2)
        wan_t_down = list(getattr(hps, 'wan_temporal_downsample', [True, True, False]))
        wan_dropout = getattr(hps, 'wan_dropout', 0.0)

        # VQ params
        latent_dim = getattr(hps, 'latent_dim', 128)
        l_bins = getattr(hps, 'l_bins', 1024)
        mu = getattr(hps, 'l_mu', 0.99)
        self.commit = getattr(hps, 'commit', 0.02)
        self.vel = getattr(hps, 'vel', 0.1)
        self.acc = getattr(hps, 'acc', 0.01)
        self.sample_length = getattr(hps, 'sample_length', 300)

        self.temporal_downsample_factor = 2 ** sum(wan_t_down)

        # Encoder (input = concat pos + rot)
        self.encoder = WanEncoder1d(
            input_dim=enc_input,
            dim=wan_dim,
            z_dim=latent_dim,
            dim_mult=wan_dim_mult,
            num_res_blocks=wan_num_res,
            temporal_downsample=wan_t_down,
            dropout=wan_dropout,
        )

        # VQ Bottleneck
        use_bottleneck = getattr(hps, 'use_bottleneck', True)
        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, latent_dim, mu, levels=1)
        else:
            self.bottleneck = NoBottleneck(levels=1)

        # Two decoders: pos and rot
        wan_t_up = wan_t_down[::-1]
        self.decoder_pos = WanDecoder1d(
            output_dim=pos_channel,
            dim=wan_dim,
            z_dim=latent_dim,
            dim_mult=wan_dim_mult,
            num_res_blocks=wan_num_res,
            temporal_upsample=wan_t_up,
            dropout=wan_dropout,
        )
        self.decoder_rot = WanDecoder1d(
            output_dim=rot_channel,
            dim=wan_dim,
            z_dim=latent_dim,
            dim_mult=wan_dim_mult,
            num_res_blocks=wan_num_res,
            temporal_upsample=wan_t_up,
            dropout=wan_dropout,
        )

        self.l_bins = l_bins
        self.latent_dim = latent_dim

    def forward(self, pos, rot):
        """pos: (B, T, pos_channel), rot: (B, T, rot_channel).
        Returns: out_pos, out_rot, loss, metrics."""
        pos_target = pos.float()
        rot_target = rot.float()
        B, T, _ = pos_target.shape

        x_cat = torch.cat([pos_target, rot_target], dim=-1)  # (B, T, C)
        x_in = x_cat.permute(0, 2, 1)  # (B, C, T)

        # Encode
        z_e = self.encoder(x_in)

        # Quantize
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck([z_e])
        z_q = xs_quantised[0]

        # Decode
        out_p = self.decoder_pos(z_q).permute(0, 2, 1)  # (B, T', pos_channel)
        out_r = self.decoder_rot(z_q).permute(0, 2, 1)  # (B, T', rot_channel)

        # Trim to original length
        if out_p.shape[1] > T:
            out_p = out_p[:, :T, :]
            out_r = out_r[:, :T, :]

        # Losses
        pos_recons = _loss_fn_l1(pos_target, out_p)
        rot_recons = _loss_fn_l1(rot_target, out_r)
        recons_loss = pos_recons + rot_recons

        velocity_loss = (
            _loss_fn_l1(out_p[:, 1:] - out_p[:, :-1],
                        pos_target[:, 1:] - pos_target[:, :-1])
            + _loss_fn_l1(out_r[:, 1:] - out_r[:, :-1],
                          rot_target[:, 1:] - rot_target[:, :-1])
        )
        acceleration_loss = (
            _loss_fn_l1(out_p[:, 2:] + out_p[:, :-2] - 2 * out_p[:, 1:-1],
                        pos_target[:, 2:] + pos_target[:, :-2] - 2 * pos_target[:, 1:-1])
            + _loss_fn_l1(out_r[:, 2:] + out_r[:, :-2] - 2 * out_r[:, 1:-1],
                          rot_target[:, 2:] + rot_target[:, :-2] - 2 * rot_target[:, 1:-1])
        )
        commit_loss = sum(commit_losses)

        loss = (recons_loss
                + self.commit * commit_loss
                + self.vel * velocity_loss
                + self.acc * acceleration_loss)

        metrics = dict(
            recons_loss=recons_loss.detach(),
            pos_recons_loss=pos_recons.detach(),
            rot_recons_loss=rot_recons.detach(),
            commit_loss=commit_loss.detach(),
            velocity_loss=velocity_loss.detach(),
            acceleration_loss=acceleration_loss.detach(),
        )
        if quantiser_metrics:
            for k, v in quantiser_metrics[0].items():
                metrics[f'vq_{k}'] = v.detach() if hasattr(v, 'detach') else v

        return out_p, out_r, loss, metrics

    def encode(self, pos, rot):
        """Returns list of code indices [(B, T_down)]."""
        x_cat = torch.cat([pos.float(), rot.float()], dim=-1)
        x_in = x_cat.permute(0, 2, 1)
        z_e = self.encoder(x_in)
        zs = self.bottleneck.encode([z_e])
        return zs

    def decode(self, zs):
        """Returns (out_pos, out_rot)."""
        xs_quantised = self.bottleneck.decode(zs, start_level=0, end_level=1)
        z_q = xs_quantised[0]
        out_p = self.decoder_pos(z_q).permute(0, 2, 1)
        out_r = self.decoder_rot(z_q).permute(0, 2, 1)
        return out_p, out_r


class WanManoVQVAE(nn.Module):
    """Left + right hand VQ-VAE wrapper with Wan backbone."""

    def __init__(self, hps_left, hps_right):
        super().__init__()
        self.vqvae_left = WanManoHandVQVAE(hps_left)
        self.vqvae_right = WanManoHandVQVAE(hps_right)

    def forward(self, left_pos, left_rot, right_pos, right_rot):
        lp, lr, loss_l, met_l = self.vqvae_left(left_pos, left_rot)
        rp, rr, loss_r, met_r = self.vqvae_right(right_pos, right_rot)
        loss = loss_l + loss_r
        metrics = {}
        for k, v in met_l.items():
            metrics[f"left_{k}"] = v
        for k, v in met_r.items():
            metrics[f"right_{k}"] = v
        return (lp, lr), (rp, rr), loss, metrics

    def encode(self, left_pos, left_rot, right_pos, right_rot):
        zl = self.vqvae_left.encode(left_pos, left_rot)
        zr = self.vqvae_right.encode(right_pos, right_rot)
        return zl, zr

    def decode(self, zl, zr):
        lp, lr = self.vqvae_left.decode(zl)
        rp, rr = self.vqvae_right.decode(zr)
        return (lp, lr), (rp, rr)
