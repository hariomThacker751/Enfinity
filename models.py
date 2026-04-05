import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        """
        Progressive Semantic Decoder Head for Vision Transformers.
        Args:
            in_channels: The embedding dimension of the backbone (e.g. 1024 for vitl14).
            out_channels: Number of segmentation classes.
            hidden_dim: The intermediate feature mapping depth.
        """
        super().__init__()
        
        # Helper function for normalized projections
        def make_projection():
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU()
            )

        # 4 independent normalized projections
        self.map_layer_1 = make_projection()
        self.map_layer_2 = make_projection()
        self.map_layer_3 = make_projection()
        self.map_layer_4 = make_projection()

        # Fusion block combining the 4 layers (4 * hidden_dim channels down to hidden_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1, bias=False),
            # GroupNorm helps stabilize statistics across the 4 drastically different depth layers
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )

        # Progressive Decoder (Upsampling)
        # Block 1: Upsamples by 2x (e.g. 1/14 -> 1/7 resolution)
        self.decoder_block_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU()
        )

        # Block 2: Upsamples by 2x (e.g. 1/7 -> 1/3.5 resolution)
        self.decoder_block_2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU()
        )
        
        # Block 3: High-Res Upsampling by 2x (e.g. 1/3.5 -> 1/1.75 resolution)
        self.decoder_block_3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 8),
            nn.GELU()
        )
        
        # Block 4: Ultra-Fidelity Upsampling by 1.75x (e.g. 1/1.75 -> 1:1 FULL resolution)
        # 14 / 2 / 2 / 2 = 1.75. This block operates on the true pixel level.
        # No GELU here: the final head is a linear projection, so we preserve the full
        # spectrum of negative and positive features for clean logit emission.
        self.decoder_block_4 = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim // 16)
            # GELU removed — no nonlinearity before the classification head
        )

        # Final Pixel-wise Projection
        self.head = nn.Conv2d(hidden_dim // 16, out_channels, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: List of 4 tensors of shape [B, C, H, W]
        """
        f1 = self.map_layer_1(features[0])
        f2 = self.map_layer_2(features[1])
        f3 = self.map_layer_3(features[2])
        f4 = self.map_layer_4(features[3])

        # Concatenate along channel dimension
        out = torch.cat([f1, f2, f3, f4], dim=1)
        
        # -------------------------------------------------------------
        # Progressive Decoder (Upsampling)
        # -------------------------------------------------------------
        # Fuse high-level semantic maps together
        out = self.fusion(out)

        # Progressive Upscaling 1 (Scale x2) -> 1/7
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = self.decoder_block_1(out)

        # Progressive Upscaling 2 (Scale x2) -> 1/3.5
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = self.decoder_block_2(out)
        
        # Progressive Upscaling 3 (Scale x2) -> 1/1.75
        out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
        out = self.decoder_block_3(out)
        
        # Progressive Upscaling 4 (Exact True Pixel Scale)
        # Instead of 1.75x which can cause floating point rounding errors,
        # we calculate the exact original target size from the patch grid.
        target_h, target_w = features[0].shape[2] * 14, features[0].shape[3] * 14
        out = F.interpolate(out, size=(target_h, target_w), mode='bilinear', align_corners=False)
        out = self.decoder_block_4(out)

        # Emit High-Resolution Semantic Logits
        logits = self.head(out)
        
        return logits
