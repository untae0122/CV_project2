import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
from model import VisionTransformer, Attention, RoPEAttention

# Configuration
DATA_DIR = './data'
RESULT_DIR = './results/50epoch'
SAVE_DIR = './results/experiments'
BATCH_SIZE = 128
DEVICE = 'cpu'
METHODS = ['sinusoidal', 'rope', 'learnable']

os.makedirs(SAVE_DIR, exist_ok=True)

def get_test_loader(batch_size=128, img_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def load_model(method, path, img_size=32):
    print(f"Loading {method} model from {path}...")
    # Initialize model with standard parameters
    model = VisionTransformer(pe_method=method, img_size=img_size)
    
    # Load state dict
    state_dict = torch.load(path, map_location=DEVICE)
    
    # Handle Learnable PE resizing if img_size is different (e.g. 48)
    if method == 'learnable' and img_size != 32:
        print("Resizing Learnable PE...")
        # LearnablePE module has 'pe' parameter, so key is 'pos_embed.pe'
        if 'pos_embed.pe' in state_dict:
            old_pe = state_dict['pos_embed.pe']
            key_name = 'pos_embed.pe'
        elif 'pos_embed' in state_dict:
             old_pe = state_dict['pos_embed']
             key_name = 'pos_embed'
        else:
            raise KeyError("Could not find pos_embed in state_dict")

        cls_pe = old_pe[:, :1, :]
        spatial_pe = old_pe[:, 1:, :]
        
        # Reshape spatial PE to 2D grid
        N_original = spatial_pe.shape[1]
        grid_size_original = int(np.sqrt(N_original))
        spatial_pe_grid = spatial_pe.transpose(1, 2).reshape(1, -1, grid_size_original, grid_size_original)
        
        # Interpolate
        grid_size_new = img_size // 4
        spatial_pe_resized = F.interpolate(spatial_pe_grid, size=(grid_size_new, grid_size_new), mode='bicubic', align_corners=False)
        
        # Flatten back
        spatial_pe_new = spatial_pe_resized.flatten(2).transpose(1, 2)
        
        # Concat CLS
        new_pe = torch.cat([cls_pe, spatial_pe_new], dim=1)
        state_dict[key_name] = new_pe

    # Handle Sinusoidal PE resizing (it's a buffer, so we need to update it if size changed)
    if method == 'sinusoidal' and img_size != 32:
         # SinusoidalPE2D re-generates PE in __init__ based on img_size, 
         # but the state_dict contains the old 'pe' buffer. We should ignore the saved buffer 
         # and let the model use its newly generated one, OR overwrite the saved buffer.
         # Since the model init already created the correct PE for new size, we can just drop 'pos_embed.pe' from state_dict
         if 'pos_embed.pe' in state_dict:
             del state_dict['pos_embed.pe']

    # Handle RoPE resizing: freqs buffers are fixed size, so we must drop them and let model use new ones
    if method == 'rope' and img_size != 32:
        keys_to_remove = [k for k in state_dict.keys() if 'freqs_cos' in k or 'freqs_sin' in k]
        for k in keys_to_remove:
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load result: {msg}")
    return model.to(DEVICE)

# --- Experiment 1: Translation Invariance ---
def test_translation_invariance(models):
    print("\n--- Experiment 1: Translation Invariance ---")
    shifts = [0, 1, 2, 3, 4]
    results = {m: [] for m in METHODS}
    
    loader = get_test_loader(BATCH_SIZE, 32)
    
    for shift in shifts:
        print(f"Testing shift: {shift}px")
        for method, model in models.items():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    
                    # Apply shift (roll)
                    # Fill with 0 (black) or edge? User said "0 or edge". Roll wraps around.
                    # Let's use slicing/padding to simulate real shift (loss of info)
                    if shift > 0:
                        # Shift right and down
                        new_inputs = torch.zeros_like(inputs)
                        new_inputs[:, :, shift:, shift:] = inputs[:, :, :-shift, :-shift]
                        inputs = new_inputs
                    
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            results[method].append(acc)
            print(f"  {method}: {acc:.2f}%")

    # Plot
    plt.figure(figsize=(8, 6))
    for method in METHODS:
        plt.plot(shifts, results[method], marker='o', label=method)
    plt.xlabel('Shift (pixels)')
    plt.ylabel('Accuracy (%)')
    plt.title('Translation Invariance Test')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'exp1_translation.png'))
    plt.close()

# --- Experiment 2: Attention Distance ---
def test_attention_distance(models):
    print("\n--- Experiment 2: Attention Distance ---")
    
    # We need to capture attention weights.
    # We will define a hook or monkey patch.
    
    attn_weights = {} # method -> list of (distance, attn_value)
    
    # Prepare distance matrix for 8x8 grid (32x32 img / 4 patch)
    grid_size = 8
    num_patches = grid_size * grid_size
    coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    # Flatten distance matrix
    flat_dists = dist_matrix.flatten()
    
    # Monkey patch forward methods to capture attention
    # We only care about the last block's attention
    
    def get_attn_hook(method_name):
        def hook(module, input, output):
            # output is the result of attention block.
            # Wait, we need the internal attention map (attn matrix).
            # Standard hook on module output gives the result of forward, which is (B, N, C).
            # We need the attention weights (B, H, N, N).
            pass
        return hook

    # Since we can't easily hook local variables, we will temporarily modify the classes or use a custom forward.
    # Let's define a wrapper or just modify the model instance's last block.
    
    loader = get_test_loader(batch_size=16, img_size=32) # Small batch
    inputs, _ = next(iter(loader))
    inputs = inputs.to(DEVICE)

    plt.figure(figsize=(8, 6))

    for method, model in models.items():
        print(f"Analyzing {method}...")
        model.eval()
        
        # Get last block
        last_block = model.blocks[-1]
        attn_module = last_block.attn
        
        # We will replace the forward method of the attention module instance
        original_forward = attn_module.forward
        
        captured_attn = []
        
        def custom_forward(x):
            # Copy-paste logic from model.py but return attn
            # This is fragile but effective for this experiment
            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            if method == 'rope':
                 # RoPE logic
                q_t = q.transpose(1, 2)
                k_t = k.transpose(1, 2)
                q_cls = q_t[:, :1, :, :]
                q_spatial = q_t[:, 1:, :, :]
                k_cls = k_t[:, :1, :, :]
                k_spatial = k_t[:, 1:, :, :]
                
                # We need access to freqs_cos/sin. They are buffers in attn_module
                q_spatial = apply_rotary_pos_emb(q_spatial, attn_module.freqs_cos, attn_module.freqs_sin)
                k_spatial = apply_rotary_pos_emb(k_spatial, attn_module.freqs_cos, attn_module.freqs_sin)
                
                q_t = torch.cat([q_cls, q_spatial], dim=1)
                k_t = torch.cat([k_cls, k_spatial], dim=1)
                q = q_t.transpose(1, 2)
                k = k_t.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * attn_module.scale
            attn = attn.softmax(dim=-1)
            
            # Capture attn (B, H, N, N)
            # We only care about spatial-spatial attention (exclude CLS)
            # CLS is index 0. Spatial is 1..N
            spatial_attn = attn[:, :, 1:, 1:].detach().cpu().numpy()
            captured_attn.append(spatial_attn)
            
            # Continue forward to not break anything (though we don't need output)
            attn = attn_module.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)
            return x

        # Bind custom forward
        # Need to import apply_rotary_pos_emb if using rope
        if method == 'rope':
            global apply_rotary_pos_emb
            from model import apply_rotary_pos_emb

        # Use types.MethodType to bind or just assign
        import types
        attn_module.forward = types.MethodType(custom_forward, attn_module) # This binds self automatically? No, custom_forward is a closure.
        # Actually, custom_forward defined above captures 'attn_module' from closure, so we can just assign it.
        # But 'self' in method call?
        # Let's just assign it as a bound method is tricky.
        # Simpler: define it taking 'self' and bind it.
        
        # Re-define to be safe
        def forward_wrapper(self, x):
            return custom_forward(x)
        
        attn_module.forward = forward_wrapper.__get__(attn_module, type(attn_module))

        # Run inference
        with torch.no_grad():
            model(inputs)
        
        # Restore
        attn_module.forward = original_forward
        
        # Process captured attention
        # captured_attn: list of (B, H, 64, 64)
        avg_attn = np.mean(np.concatenate(captured_attn, axis=0), axis=(0, 1)) # (64, 64)
        
        # Bin by distance
        flat_attn = avg_attn.flatten()
        
        # Calculate average attention per distance
        # Round distances to integers for binning
        rounded_dists = np.round(flat_dists).astype(int)
        max_dist = rounded_dists.max()
        
        dist_bins = []
        attn_bins = []
        
        for d in range(max_dist + 1):
            mask = (rounded_dists == d)
            if mask.sum() > 0:
                dist_bins.append(d)
                attn_bins.append(flat_attn[mask].mean())
        
        plt.plot(dist_bins, attn_bins, label=method)

    plt.xlabel('Pixel Distance')
    plt.ylabel('Average Attention Weight')
    plt.title('Attention Distance Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, 'exp2_attention_distance.png'))
    plt.close()

# --- Experiment 3: Resolution Robustness ---
def test_resolution_robustness():
    print("\n--- Experiment 3: Resolution Robustness ---")
    
    # Load models with img_size=48
    img_size = 48
    loader = get_test_loader(BATCH_SIZE, img_size)
    
    results = {}
    
    for method in METHODS:
        path = os.path.join(RESULT_DIR, f'vit_{method}.pth')
        if not os.path.exists(path):
            print(f"Skipping {method}, model not found.")
            continue
            
        model = load_model(method, path, img_size=img_size)
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        results[method] = acc
        print(f"  {method} (48x48): {acc:.2f}%")
    
    # Plot bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel('Accuracy (%)')
    plt.title(f'Resolution Robustness (32x32 -> {img_size}x{img_size})')
    plt.ylim(0, 100)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.savefig(os.path.join(SAVE_DIR, 'exp3_resolution.png'))
    plt.close()

def main():
    # Load 32x32 models for Exp 1 & 2
    models_32 = {}
    for method in METHODS:
        path = os.path.join(RESULT_DIR, f'vit_{method}.pth')
        if os.path.exists(path):
            models_32[method] = load_model(method, path, img_size=32)
        else:
            print(f"Warning: {path} not found.")
    
    if models_32:
        test_translation_invariance(models_32)
        test_attention_distance(models_32)
    
    # Exp 3 loads its own models (resized)
    test_resolution_robustness()
    
    print(f"\nAll experiments finished. Results saved to {SAVE_DIR}")

if __name__ == '__main__':
    main()
