"""
Bidirectional 1D Encoder/Decoder based on Wan2.2 VAE architecture.
Original source: https://github.com/Wan-Video/Wan2.2 (Alibaba Wan Team)
Modified: causal conv -> standard bidirectional conv, removed streaming cache.
Operates on (B, C, T) tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm1d(in_dim),
            nn.SiLU(),
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
            RMSNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)


class AvgDown1D(nn.Module):
    """Downsample by factor_t using reshape + group mean (from Wan2.2)."""
    def __init__(self, in_channels, out_channels, factor_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor = self.factor_t
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = F.pad(x, (0, pad_t))
        B, C, T = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t)
        x = x.mean(dim=2)
        return x


class DupUp1D(nn.Module):
    """Upsample by factor_t using repeat interleave + reshape (from Wan2.2)."""
    def __init__(self, in_channels, out_channels, factor_t):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor = self.factor_t
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t, x.size(2))
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks,
                 temporal_downsample=False):
        super().__init__()
        if temporal_downsample:
            self.avg_shortcut = AvgDown1D(in_dim, out_dim, factor_t=2)
        else:
            self.avg_shortcut = None

        layers = []
        dim = in_dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(dim, out_dim, dropout))
            dim = out_dim
        if temporal_downsample:
            layers.append(nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x_skip = x
        for layer in self.layers:
            x = layer(x)
        if self.avg_shortcut is not None:
            return x + self.avg_shortcut(x_skip)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks,
                 temporal_upsample=False):
        super().__init__()
        if temporal_upsample:
            self.dup_shortcut = DupUp1D(in_dim, out_dim, factor_t=2)
        else:
            self.dup_shortcut = None

        layers = []
        dim = in_dim
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(dim, out_dim, dropout))
            dim = out_dim
        if temporal_upsample:
            layers.append(nn.ConvTranspose1d(out_dim, out_dim, 4, stride=2, padding=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x_skip = x
        for layer in self.layers:
            x = layer(x)
        if self.dup_shortcut is not None:
            return x + self.dup_shortcut(x_skip)
        return x


class WanEncoder1d(nn.Module):
    """Bidirectional 1D encoder with Wan-style architecture.

    Args:
        input_dim: input channels (e.g. 60)
        dim: base channel width
        z_dim: output latent channels
        dim_mult: channel multipliers per stage
        num_res_blocks: residual blocks per stage
        temporal_downsample: list of bool per stage
        dropout: dropout rate
    """
    def __init__(self, input_dim, dim=128, z_dim=128,
                 dim_mult=(1, 2, 4, 4), num_res_blocks=2,
                 temporal_downsample=(True, True, False), dropout=0.0):
        super().__init__()
        dims = [dim * m for m in (1,) + tuple(dim_mult)]

        self.conv_in = nn.Conv1d(input_dim, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temporal_downsample[i] if i < len(temporal_downsample) else False
            self.down_blocks.append(
                DownBlock(d_in, d_out, dropout, num_res_blocks, t_down))

        last_dim = dims[-1]
        self.mid = nn.Sequential(
            ResidualBlock(last_dim, last_dim, dropout),
            RMSNorm1d(last_dim),
            nn.Conv1d(last_dim, last_dim, 1),
            ResidualBlock(last_dim, last_dim, dropout),
        )

        self.head = nn.Sequential(
            RMSNorm1d(last_dim),
            nn.SiLU(),
            nn.Conv1d(last_dim, z_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class WanDecoder1d(nn.Module):
    """Bidirectional 1D decoder with Wan-style architecture.

    Args:
        output_dim: output channels (e.g. 60)
        dim: base channel width
        z_dim: input latent channels
        dim_mult: channel multipliers per stage (will be reversed)
        num_res_blocks: residual blocks per stage
        temporal_upsample: list of bool per stage (reversed from encoder)
        dropout: dropout rate
    """
    def __init__(self, output_dim, dim=128, z_dim=128,
                 dim_mult=(1, 2, 4, 4), num_res_blocks=2,
                 temporal_upsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * m for m in (dim_mult[-1],) + tuple(dim_mult[::-1])]

        self.conv_in = nn.Conv1d(z_dim, dims[0], 3, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            RMSNorm1d(dims[0]),
            nn.Conv1d(dims[0], dims[0], 1),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        self.up_blocks = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temporal_upsample[i] if i < len(temporal_upsample) else False
            self.up_blocks.append(
                UpBlock(d_in, d_out, dropout, num_res_blocks + 1, t_up))

        last_dim = dims[-1]
        self.head = nn.Sequential(
            RMSNorm1d(last_dim),
            nn.SiLU(),
            nn.Conv1d(last_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.head(x)
        return x
