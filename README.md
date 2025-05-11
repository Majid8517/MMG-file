import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Tuple, Optional

class LearnableMPP(nn.Module):
    """Multi-Planar Projection with trainable weights"""
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.projection_weights = nn.Parameter(torch.ones(3, 64)  # 3 planes x 64 slices
        self.conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        # Generate orthogonal projections
        axial = reduce(x * F.softmax(self.projection_weights[0], dim=0)[None,:,None,None], 
                      'b c d h w -> b c h w', 'sum')
        sagittal = reduce(x.permute(0,1,3,2,4) * F.softmax(self.projection_weights[1], dim=0)[None,:,None,None],
                         'b c h d w -> b c h w', 'sum')
        coronal = reduce(x.permute(0,1,4,2,3) * F.softmax(self.projection_weights[2], dim=0)[None,:,None,None],
                        'b c w d h -> b c h w', 'sum')
        
        return self.conv(torch.cat([axial, sagittal, coronal], dim=1))

class DeformableSTN(nn.Module):
    """Deformable Spatial Transformer with Edge Awareness"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(x) * 5  # Constrain deformation magnitude
        edge_mask = self.edge_detector(x)
        grid = F.affine_grid(
            torch.eye(2,3)[None].repeat(x.size(0),1,1).to(x.device), 
            x.size()
        )
        grid = grid + offsets.permute(0,2,3,1) * edge_mask.permute(0,2,3,1)
        return F.grid_sample(x, grid, padding_mode='border', align_corners=False)

class CrossModalAttention(nn.Module):
    """Memory-Efficient Cross Attention"""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x2d.shape
        x2d = rearrange(x2d, 'b c h w -> b (h w) c')
        x3d = reduce(x3d, 'b c d h w -> b (h w) c', 'max')  # Reduce 3D->2D
        
        q = rearrange(self.to_q(x2d), 'b n (h d) -> b h n d', h=self.heads)
        k, v = rearrange(self.to_kv(x3d), 'b n (kv h d) -> kv b h n d', 
                        kv=2, h=self.heads)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        
        out = rearrange(attn @ v, 'b h n d -> b n (h d)')
        return self.proj(out).view(B, C, H, W)

class MMG_SiamNet(nn.Module):
    def __init__(self, 
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 mc_samples: int = 10):
        super().__init__()
        # Encoders
        self.encoder_2d = ResNet2D()
        self.encoder_3d = nnUNet3D()
        
        # Alignment
        self.mpp = LearnableMPP()
        self.stn = DeformableSTN(256)
        self.attn = CrossModalAttention(256)
        
        # Decoder
        self.decoder = PANDecoder()
        
        # Heads
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Uncertainty
        self.dropout = nn.Dropout2d(dropout_rate)
        self.mc_samples = mc_samples
        
    def forward(self, 
               x2d: torch.Tensor, 
               x3d: torch.Tensor
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Feature extraction
        f2d = self.encoder_2d(x2d)  # [B,256,H,W]
        f3d = self.encoder_3d(x3d)  # [B,256,D,H,W]
        
        # Cross-modal alignment
        f3d_proj = self.mpp(f3d)
        f3d_aligned = self.stn(f3d_proj)
        
        # Attention fusion
        fused = self.attn(f2d, f3d_aligned)
        
        # Decoding
        features = self.decoder(fused)
        
        # Output
        if self.training:
            return self.seg_head(features)
        else:
            # MC Dropout for uncertainty
            outputs = torch.stack([
                self.seg_head(self.dropout(features)) 
                for _ in range(self.mc_samples)
            ])
            return outputs.mean(0), outputs.var(0)

# Helper Modules (implemented separately)
class ResNet2D(nn.Module): ...
class nnUNet3D(nn.Module): ...
class PANDecoder(nn.Module): ...# MMG-file
