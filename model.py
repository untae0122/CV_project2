import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, H/P, W/P) -> (B, E, N) -> (B, N, E)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# --- Positional Encodings ---

class SinusoidalPE2D(nn.Module):
    def __init__(self, embed_dim, n_patches, img_size, patch_size):
        super().__init__()
        self.n_patches = n_patches
        # 2D Sinusoidal PE
        grid_size = img_size // patch_size
        num_spatial_patches = grid_size * grid_size
        
        pe = torch.zeros(num_spatial_patches, embed_dim)
        
        # Create a grid of coordinates
        y_pos, x_pos = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        y_pos = y_pos.flatten()
        x_pos = x_pos.flatten()

        # We use half of dimensions for x and half for y
        d_model_half = embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        pe[:, 0:d_model_half:2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 1:d_model_half:2] = torch.cos(x_pos.unsqueeze(1) * div_term)
        pe[:, d_model_half::2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, d_model_half+1::2] = torch.cos(y_pos.unsqueeze(1) * div_term)
        
        # Add CLS token PE (zeros) at the beginning
        cls_pe = torch.zeros(1, embed_dim)
        pe = torch.cat([cls_pe, pe], dim=0) # (N+1, E)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, N+1, E)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LearnablePE(nn.Module):
    def __init__(self, embed_dim, n_patches):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe

# --- RoPE Implementation ---
def apply_rotary_pos_emb(x, freqs_cos, freqs_sin):
    # x: (B, N, H, D)
    # freqs_cos, freqs_sin: (1, N, 1, D) - broadcastable
    return (x * freqs_cos) + (rotate_half(x) * freqs_sin)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., 
                 img_size=32, patch_size=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Precompute RoPE frequencies
        grid_size = img_size // patch_size
        self.head_dim = head_dim
        
        # Generate frequencies
        theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        
        # 2D grid
        y_pos, x_pos = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
        y_pos = y_pos.flatten()
        x_pos = x_pos.flatten()
        
        # We apply RoPE to half of the head dim for x and half for y effectively?
        # Standard RoPE is usually 1D or we need to interleave.
        # For 2D images, a common strategy is to use half dims for X and half for Y.
        
        freqs_x = torch.outer(x_pos, theta)
        freqs_y = torch.outer(y_pos, theta)
        
        # We repeat to match head_dim size if we split it.
        # Strategy: First half of head_dim encodes X, second half encodes Y.
        # This assumes head_dim is even.
        
        freqs_x_complex = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_y_complex = torch.polar(torch.ones_like(freqs_y), freqs_y)
        
        # Actually, simpler implementation for RoPE:
        # Just compute cos and sin.
        # Let's split head_dim into two halves.
        
        half_dim = head_dim // 2
        theta_half = 10000.0 ** (-torch.arange(0, half_dim, 2).float() / half_dim)
        
        freqs_x = torch.outer(x_pos, theta_half) # (N, half_dim/2)
        freqs_y = torch.outer(y_pos, theta_half) # (N, half_dim/2)
        
        # We need (N, head_dim) total.
        # We construct it such that we have pairs for rotation.
        # Let's construct full (N, head_dim) rotation angles.
        # [x_freqs, x_freqs, y_freqs, y_freqs] pattern is one way, or just concat.
        
        # Let's use the strategy:
        # features [0:half_dim] use x_pos
        # features [half_dim:] use y_pos
        
        freqs_x_expanded = freqs_x.repeat_interleave(2, dim=1) # (N, half_dim)
        freqs_y_expanded = freqs_y.repeat_interleave(2, dim=1) # (N, half_dim)
        
        freqs = torch.cat([freqs_x_expanded, freqs_y_expanded], dim=1) # (N, head_dim)
        
        self.register_buffer("freqs_cos", freqs.cos().unsqueeze(0).unsqueeze(2)) # (1, N, 1, D)
        self.register_buffer("freqs_sin", freqs.sin().unsqueeze(0).unsqueeze(2))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, D)

        # Apply RoPE to Q and K
        # Transpose to (B, N, H, D) for easier broadcasting
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        
        # Split CLS and Spatial
        # Assuming CLS is at index 0
        q_cls = q_t[:, :1, :, :]
        q_spatial = q_t[:, 1:, :, :]
        k_cls = k_t[:, :1, :, :]
        k_spatial = k_t[:, 1:, :, :]
        
        # Apply RoPE to spatial only
        # freqs_cos/sin are (1, N_spatial, 1, D)
        q_spatial = apply_rotary_pos_emb(q_spatial, self.freqs_cos, self.freqs_sin)
        k_spatial = apply_rotary_pos_emb(k_spatial, self.freqs_cos, self.freqs_sin)
        
        # Concat back
        q_t = torch.cat([q_cls, q_spatial], dim=1)
        k_t = torch.cat([k_cls, k_spatial], dim=1)
        
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- Standard Attention (for non-RoPE) ---
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rope=False, img_size=32, patch_size=4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                      img_size=img_size, patch_size=patch_size)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=192, depth=9,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 pe_method='sinusoidal'): # pe_method: 'sinusoidal', 'learnable', 'rope'
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.pe_method = pe_method

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Encoding
        if pe_method == 'sinusoidal':
            self.pos_embed = SinusoidalPE2D(embed_dim, num_patches + 1, img_size, patch_size)
        elif pe_method == 'learnable':
            self.pos_embed = LearnablePE(embed_dim, num_patches + 1)
        elif pe_method == 'rope':
            self.pos_embed = nn.Identity() # RoPE is applied in Attention
        else:
            raise ValueError(f"Unknown PE method: {pe_method}")

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, use_rope=(pe_method == 'rope'),
                img_size=img_size, patch_size=patch_size
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Init weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pe_method != 'rope':
            x = self.pos_embed(x)
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0] # CLS token
        x = self.head(x)
        return x
