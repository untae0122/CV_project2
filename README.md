# Analysis and Design of Positional Encoding Methods in Vision Transformers

This project implements a Vision Transformer (ViT) from scratch on CIFAR-10 and compares three positional encoding methods:
1.  **Baseline:** 2D Sinusoidal Positional Encoding
2.  **RoPE:** Rotary Positional Embedding
3.  **Proposed:** Learnable Positional Encoding

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Train and Evaluate

You can run the training script `main.py` with the following arguments:

-   `--pe_method`: Choose the positional encoding method (`sinusoidal`, `rope`, `learnable`, or `all`). Default is `all`.
-   `--epochs`: Number of training epochs. Default is 50.
-   `--batch_size`: Batch size. Default is 128.
-   `--lr`: Learning rate. Default is 1e-3.
-   `--save_dir`: Directory to save results and models. Default is `./results`.

**Example: Run all methods (recommended)**
```bash
python main.py --pe_method all --epochs 50 --batch_size 128
```

**Example: Run specific method**
```bash
python main.py --pe_method rope --epochs 50
```

**Example: Run specific gpu**
CUDA_VISIBLE_DEVICES=1 python main.py --pe_method all --epochs 50 --batch_size 128

## Results

The script will save:
-   Model checkpoints (`.pth`) in the `results/` directory.
-   Loss and accuracy plots (`.png`) in the `results/` directory.
-   A JSON file (`results.json`) containing all metrics.
