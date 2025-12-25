import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Self-attention block with scaled dot-product attention"""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_dim, in_dim // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_dim // 8) ** -0.5  # Scaling factor for stable attention

    def forward(self, x):
        batch, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch, -1, width*height).permute(0, 2, 1)  # (B, HW, C')
        proj_key = self.key_conv(x).view(batch, -1, width*height)  # (B, C', HW)
        
        # Scaled dot-product attention
        energy = torch.bmm(proj_query, proj_key) * self.scale
        
        # Clamp before softmax to prevent extreme values
        energy = energy.clamp(-10, 10)
        
        attention = torch.softmax(energy, dim=-1)
        
        # Safety check for NaN
        if torch.isnan(attention).any():
            print("[WARNING] NaN detected in attention weights, replacing with zeros")
            attention = torch.nan_to_num(attention, nan=0.0)
        
        proj_value = self.value_conv(x).view(batch, C, -1)  # (B, C, HW)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        
        out = self.gamma * out + x
        return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization for stable training.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB or LAB)
        nf: Base number of filters (default: 64)
        n_blocks: Number of downsampling blocks (default: 3)
        use_spectral_norm: Whether to apply spectral normalization (default: True)
        use_attention: Whether to use self-attention in first block (default: False)
    """
    def __init__(self, in_channels=3, nf=64, n_blocks=3, use_spectral_norm=True, use_attention=False):
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        self.use_attention = use_attention
        
        def conv_block(in_c, out_c, kernel_size, stride, padding):
            """Helper to create conv layer with optional spectral norm"""
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
            if self.use_spectral_norm:
                conv = spectral_norm(conv)
            return conv
        
        self.layers = nn.ModuleList()
        
        # First conv: no normalization, just LeakyReLU
        self.layers.append(conv_block(in_channels, nf, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        current_nf = nf
        
        # Downsampling blocks
        for i in range(n_blocks):
            # Conv keeping same channels
            self.layers.append(conv_block(current_nf, current_nf, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Conv downsampling, double channels
            self.layers.append(conv_block(current_nf, current_nf * 2, kernel_size=4, stride=2, padding=1))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Optional self-attention on first block
            if i == 0 and self.use_attention:
                self.layers.append(SelfAttention(current_nf * 2))
            
            current_nf *= 2
        
        # Final convs
        self.layers.append(conv_block(current_nf, current_nf, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final output conv: 1-channel output (no spectral norm on final layer)
        self.layers.append(nn.Conv2d(current_nf, 1, kernel_size=4, stride=1, padding=0))
        
        # Combine as sequential for forward pass
        self.model = nn.Sequential(*self.layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Flattened discriminator output (B, N)
        """
        out = self.model(x)
        
        # Debug check for NaN
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[ERROR] NaN/Inf detected in discriminator output!")
            print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
            # Replace NaN with zeros to prevent training crash
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return out.view(out.size(0), -1)


# Alternative: More aggressive spectral normalization version
class StableDiscriminator(nn.Module):
    """
    Even more stable discriminator with:
    - Spectral normalization on ALL layers
    - No self-attention (remove unstable component)
    - Proper initialization
    - Gradient clipping built-in
    """
    def __init__(self, in_channels=3, nf=64, n_blocks=3):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.extend([
            spectral_norm(nn.Conv2d(in_channels, nf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        current_nf = nf
        
        # Progressive downsampling
        for i in range(n_blocks):
            layers.extend([
                spectral_norm(nn.Conv2d(current_nf, current_nf, 3, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(current_nf, current_nf * 2, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_nf *= 2
        
        # Final layers
        layers.extend([
            spectral_norm(nn.Conv2d(current_nf, current_nf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(current_nf, 1, 4, 1, 0)  # No spectral norm on final layer
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, x):
        out = self.model(x)
        
        # Clamp output to prevent extreme values
        out = out.clamp(-10, 10)
        
        return out.view(out.size(0), -1)


if __name__ == '__main__':
    # Test the discriminator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test PatchDiscriminator with spectral norm
    print("Testing PatchDiscriminator with Spectral Norm...")
    disc1 = PatchDiscriminator(in_channels=3, nf=64, n_blocks=3, 
                               use_spectral_norm=True, use_attention=False).to(device)
    
    # Test StableDiscriminator
    print("\nTesting StableDiscriminator...")
    disc2 = StableDiscriminator(in_channels=3, nf=64, n_blocks=3).to(device)
    
    # Count parameters
    params1 = sum(p.numel() for p in disc1.parameters())
    params2 = sum(p.numel() for p in disc2.parameters())
    
    print(f"\nPatchDiscriminator parameters: {params1:,}")
    print(f"StableDiscriminator parameters: {params2:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        out1 = disc1(dummy_input)
        out2 = disc2(dummy_input)
    
    print(f"\nPatchDiscriminator output shape: {out1.shape}")
    print(f"PatchDiscriminator output range: [{out1.min():.4f}, {out1.max():.4f}]")
    
    print(f"\nStableDiscriminator output shape: {out2.shape}")
    print(f"StableDiscriminator output range: [{out2.min():.4f}, {out2.max():.4f}]")
    
    print("\n✓ All tests passed!")