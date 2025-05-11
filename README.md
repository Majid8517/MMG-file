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
import torch
import torch.nn as nn
import torch.nn.functional as F

class nnUNet3DBlock(nn.Module):
    """Modified 3D nnUNet block with adaptive depth-wise convolutions"""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 num_groups: int = 8):
        super().__init__()
        
        # Depth-wise separable convolution for efficiency
        self.dw_conv = nn.Conv3d(in_channels, in_channels, 
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=tuple(k//2 for k in kernel_size),
                                groups=in_channels)
        
        # Point-wise convolution
        self.pw_conv = nn.Conv3d(in_channels, out_channels, 
                                kernel_size=1)
        
        # Adaptive normalization
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.LeakyReLU(0.01)  # Better for medical images
        
        # Residual connection
        self.residual = nn.Sequential()
        if in_channels != out_channels or any(s > 1 for s in stride):
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        # Depth-wise separable convolution
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        
        # Normalization and activation
        x = self.norm(x)
        x = self.act(x + residual)
        
        return x

class nnUNet3D(nn.Module):
    """Complete 3D nnUNet encoder adapted for MMG-SiamNet"""
    def __init__(self, 
                 in_channels: int = 1,
                 base_channels: int = 32,
                 num_blocks: int = 4):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 
                     kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.01)
        )
        
        # Downsampling blocks
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = current_channels * 2 if i < num_blocks - 1 else current_channels
            
            self.blocks.append(
                nn.Sequential(
                    nnUNet3DBlock(current_channels, out_channels,
                                stride=(1, 2, 2) if i == 0 else (2, 2, 2)),
                    nnUNet3DBlock(out_channels, out_channels)
                )
            )
            current_channels = out_channels
            
        # Feature refinement
        self.final_conv = nn.Conv3d(current_channels, current_channels,
                                   kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        x = self.init_conv(x)
        
        for block in self.blocks:
            x = block(x)
            features.append(x)  # For skip connections
            
        return self.final_conv(x), features  # Return both final features and skips
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
       def __init__(self):
        super().__init__()
        self.encoder_3d = nnUNet3D(in_channels=1, base_channels=32)
        
    def forward(self, x3d):
        # x3d: [B, 1, 64, 64, 64]
        features_3d, skips_3d = self.encoder_3d(x3d) 
        # features_3d: [B, 256, 4, 4, 4]
        # skips_3d: List of features at different scales
        return features_3d 
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
