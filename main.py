import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import json

from model import VisionTransformer
from utils import set_seed, plot_metrics, plot_comparison

def get_args():
    parser = argparse.ArgumentParser(description='ViT Positional Encoding Analysis')
    parser.add_argument('--pe_method', type=str, default='all', choices=['sinusoidal', 'rope', 'learnable', 'all'],
                        help='Positional encoding method to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs') # Reduced default for faster testing, user can increase
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./results', help='Result save directory')
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return running_loss / total, 100. * correct / total

def main():
    args = get_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data Preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    methods = ['sinusoidal', 'rope', 'learnable'] if args.pe_method == 'all' else [args.pe_method]
    
    all_results = {}

    for method in methods:
        print(f"\nTraining with {method} PE...")
        model = VisionTransformer(pe_method=method).to(device)
        
        # Using AdamW as it's standard for ViTs
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_losses = []
        val_accs = []
        
        start_time = time.time()
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, testloader, criterion, device)
            scheduler.step()
            
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f}s")
        
        # Save metrics
        plot_metrics(train_losses, val_accs, method, save_dir=args.save_dir)
        all_results[method] = {
            'train_loss': train_losses,
            'val_acc': val_accs,
            'final_val_acc': val_accs[-1],
            'training_time': total_time
        }
        
        # Save model
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'vit_{method}.pth'))

    # Save all results to json
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
        
    if len(methods) > 1:
        plot_comparison(all_results, save_dir=args.save_dir)
        print("\nComparison plot saved.")

if __name__ == '__main__':
    main()
