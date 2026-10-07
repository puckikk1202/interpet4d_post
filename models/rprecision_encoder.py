"""Contrastive encoder pair for R-Precision evaluation.

Two independent EncoderConvBlock encoders map pet motion and human motion
(MANO or SMPL) to a shared L2-normalized embedding space. Trained with
symmetric InfoNCE (CLIP-style) loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encdec import EncoderConvBlock


class RPrecisionEncoder(nn.Module):
    """Dual-encoder contrastive model for R-Precision.

    Args:
        dim_a (int): Input dimension for encoder A (pet motion, e.g. 60).
        dim_b (int): Input dimension for encoder B (mano 414 or smpl 207).
        emb_dim (int): Shared embedding dimension (default 256).
        width (int): Hidden width in ResNet blocks.
        depth (int): Depth of each ResNet block.
        downs_t (list[int]): Number of downsampling stages.
        strides_t (list[int]): Stride per downsampling stage.
        m_conv (float): Conv multiplier (default 1.0).
        dilation_growth_rate (int): Dilation growth (default 3).
        dropout (float): Dropout rate before projection (default 0.0).
    """

    def __init__(self, dim_a, dim_b, emb_dim=256, width=512, depth=4,
                 downs_t=(2,), strides_t=(2,), m_conv=1.0,
                 dilation_growth_rate=3, dropout=0.0):
        super().__init__()

        # Build encoder A (pet)
        enc_a_blocks = []
        in_ch = dim_a
        for down_t, stride_t in zip(downs_t, strides_t):
            enc_a_blocks.append(EncoderConvBlock(
                input_emb_width=in_ch,
                output_emb_width=width,
                down_t=down_t,
                stride_t=stride_t,
                width=width,
                depth=depth,
                m_conv=m_conv,
                dilation_growth_rate=dilation_growth_rate,
            ))
            in_ch = width
        self.enc_a = nn.Sequential(*enc_a_blocks)
        self.proj_a = nn.Linear(width, emb_dim)

        # Build encoder B (mano or smpl)
        enc_b_blocks = []
        in_ch = dim_b
        for down_t, stride_t in zip(downs_t, strides_t):
            enc_b_blocks.append(EncoderConvBlock(
                input_emb_width=in_ch,
                output_emb_width=width,
                down_t=down_t,
                stride_t=stride_t,
                width=width,
                depth=depth,
                m_conv=m_conv,
                dilation_growth_rate=dilation_growth_rate,
            ))
            in_ch = width
        self.enc_b = nn.Sequential(*enc_b_blocks)
        self.proj_b = nn.Linear(width, emb_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable temperature (log-space, initialized to CLIP default 0.07)
        self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

    def encode_a(self, x_a):
        """Encode pet motion to L2-normalized embedding.

        Args:
            x_a: (B, T, dim_a)
        Returns:
            z_a: (B, emb_dim)
        """
        h = self.enc_a(x_a.permute(0, 2, 1))  # (B, width, T')
        h = h.mean(dim=-1)                      # GAP -> (B, width)
        h = self.dropout(h)
        z = F.normalize(self.proj_a(h), dim=-1)
        return z

    def encode_b(self, x_b):
        """Encode human motion to L2-normalized embedding.

        Args:
            x_b: (B, T, dim_b)
        Returns:
            z_b: (B, emb_dim)
        """
        h = self.enc_b(x_b.permute(0, 2, 1))  # (B, width, T')
        h = h.mean(dim=-1)                      # GAP -> (B, width)
        h = self.dropout(h)
        z = F.normalize(self.proj_b(h), dim=-1)
        return z

    def forward(self, x_a, x_b):
        """Forward pass returning both embeddings.

        Args:
            x_a: (B, T, dim_a) pet motion
            x_b: (B, T, dim_b) human motion (mano or smpl)
        Returns:
            z_a: (B, emb_dim) L2-normalized pet embedding
            z_b: (B, emb_dim) L2-normalized human embedding
        """
        return self.encode_a(x_a), self.encode_b(x_b)

    def info_nce_loss(self, z_a, z_b):
        """Symmetric InfoNCE loss (CLIP-style).

        Args:
            z_a: (B, emb_dim) L2-normalized
            z_b: (B, emb_dim) L2-normalized
        Returns:
            loss: scalar
        """
        temperature = self.log_temperature.exp()
        logits = z_a @ z_b.T * temperature.reciprocal()  # (B, B)
        labels = torch.arange(z_a.shape[0], device=z_a.device)
        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2
        return loss
