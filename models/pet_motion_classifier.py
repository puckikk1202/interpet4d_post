import torch
import torch.nn as nn
from archived.encdec import EncoderConvBlock


class PetMotionClassifier(nn.Module):
    """Dog identity classifier on pet motion sequences.

    Uses EncoderConvBlock (1D ResNet) as backbone, global average pooling,
    then a linear head for classification.  The penultimate-layer features
    (after GAP, before FC) serve as a learned feature space for FID.

    Args:
        config: EasyDict with keys:
            input_dim (int): motion feature dim (default 60).
            feat_dim (int): output feature dim from encoder (default 512).
            num_classes (int): number of dog identities.
            width (int): hidden width in ResNet blocks.
            depth (int): depth of each ResNet block.
            downs_t (list[int]): number of downsampling stages.
            strides_t (list[int]): stride per downsampling stage.
            m_conv (float): conv multiplier (default 1.0).
            dilation_growth_rate (int): dilation growth (default 3).
    """

    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim      # 60
        feat_dim = config.feat_dim         # 512
        num_classes = config.num_classes
        width = config.get('width', 512)
        depth = config.get('depth', 4)
        downs_t = config.get('downs_t', [2])
        strides_t = config.get('strides_t', [2])
        m_conv = config.get('m_conv', 1.0)
        dilation_growth_rate = config.get('dilation_growth_rate', 3)

        # Build encoder: stack one EncoderConvBlock per downsampling level
        blocks = []
        in_ch = input_dim
        for i, (down_t, stride_t) in enumerate(zip(downs_t, strides_t)):
            blocks.append(EncoderConvBlock(
                input_emb_width=in_ch,
                output_emb_width=feat_dim,
                down_t=down_t,
                stride_t=stride_t,
                width=width,
                depth=depth,
                m_conv=m_conv,
                dilation_growth_rate=dilation_growth_rate,
            ))
            in_ch = feat_dim
        self.encoder = nn.Sequential(*blocks)

        # Classification head
        self.head = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x):
        """Extract penultimate-layer features (for FID computation).

        Args:
            x: (B, T, input_dim) motion tensor.
        Returns:
            feats: (B, feat_dim) feature vectors.
        """
        # (B, T, C) -> (B, C, T)
        h = x.permute(0, 2, 1)
        h = self.encoder(h)           # (B, feat_dim, T')
        feats = h.mean(dim=-1)        # global average pooling -> (B, feat_dim)
        return feats

    def forward(self, x):
        """Forward pass returning logits and features.

        Args:
            x: (B, T, input_dim) motion tensor.
        Returns:
            logits: (B, num_classes)
            feats: (B, feat_dim)
        """
        feats = self.extract_features(x)
        logits = self.head(feats)
        return logits, feats
