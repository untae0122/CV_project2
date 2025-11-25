import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_metrics(train_losses, val_accs, method_name, save_dir='.'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title(f'{method_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title(f'{method_name} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method_name}_metrics.png'))
    plt.close()

def plot_comparison(results, save_dir='.'):
    # results: dict {method_name: {'train_loss': [], 'val_acc': []}}
    
    plt.figure(figsize=(12, 5))
    
    # Compare Loss
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        plt.plot(data['train_loss'], label=method)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Compare Accuracy
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        plt.plot(data['val_acc'], label=method)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_metrics.png'))
    plt.close()
