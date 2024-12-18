# train.py
"""
Optimized training script for A100 GPU
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve)
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from metricstracker import MetricsTracker
from model import get_model
from dataloader import get_dataloaders

def create_output_dirs(model_name):
    """Create output directories for storing model results"""
    output_dirs = {
        'base': f'run/{model_name}',
        'models': f'run/{model_name}/models',
        'metrics': f'run/{model_name}/metrics',
        'plots': f'run/{model_name}/plots'
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dirs

def train_model(model_name, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Train the model with increased regularization"""
    output_dirs = create_output_dirs(model_name)
    metrics_tracker = MetricsTracker(output_dirs)
    
    # Enable cuDNN benchmarking
    cudnn.benchmark = True
    
    # Initialize model
    model = get_model(model_name, num_classes=2)
    model = model.to(device)
    
    # Initialize loss and optimizer with stronger regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,  # Increased from 0.01
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )
    
    # Early stopping setup
    early_stopping_patience = 5
    no_improve_count = 0
    best_val_loss = float('inf')
    best_val_auc = 0
    
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_metrics = {
            'train_loss': train_loss / len(train_loader),
            'train_auc': roc_auc_score(train_labels, train_preds),
            'train_acc': accuracy_score(train_labels, train_preds),
            'train_precision': precision_score(train_labels, train_preds),
            'train_recall': recall_score(train_labels, train_preds),
            'train_f1': f1_score(train_labels, train_preds)
        }
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        
        # Calculate validation metrics
        val_metrics = {
            'val_loss': val_loss,
            'val_auc': roc_auc_score(val_labels, val_preds),
            'val_acc': accuracy_score(val_labels, val_preds),
            'val_precision': precision_score(val_labels, val_preds),
            'val_recall': recall_score(val_labels, val_preds),
            'val_f1': f1_score(val_labels, val_preds)
        }
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Save best model based on AUC
        if val_metrics['val_auc'] > best_val_auc:
            best_val_auc = val_metrics['val_auc']
            torch.save(model.state_dict(), 
                      f"{output_dirs['models']}/{model_name}_best.pt")
        
        # Update metrics tracker
        metrics = {**train_metrics, **val_metrics}
        metrics_tracker.update(epoch, metrics)
        
        # Plot ROC curve and confusion matrix periodically
        if (epoch + 1) % 5 == 0:
            fpr, tpr, _ = roc_curve(val_labels, val_preds)
            metrics_tracker.plot_roc_curve(fpr, tpr, val_metrics['val_auc'])
            metrics_tracker.plot_confusion_matrix(val_labels, val_preds)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f"Train - Loss: {metrics['train_loss']:.4f}, AUC: {metrics['train_auc']:.4f}, "
              f"Acc: {metrics['train_acc']:.4f}, F1: {metrics['train_f1']:.4f}")
        print(f"Val - Loss: {metrics['val_loss']:.4f}, AUC: {metrics['val_auc']:.4f}, "
              f"Acc: {metrics['val_acc']:.4f}, F1: {metrics['val_f1']:.4f}")
        
        # Early stopping
        if no_improve_count >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final metrics and plots
    metrics_tracker.save_metrics()
    metrics_tracker.plot_metrics()
    
    return model, metrics_tracker

def main():
    parser = argparse.ArgumentParser(description='Train histopathology classification model')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet', 'mamba'],
                      help='Model architecture to use (vit or resnet or mamba)')
    parser.add_argument('--data_root', type=str, default='data',
                      help='Root directory for the dataset')
    parser.add_argument('--batch_size', type=int, default=256,  # Increased batch size
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=8,  # Adjusted workers
                      help='Number of worker processes for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                      help='Pin memory for faster data transfer to GPU')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training {args.model.upper()} model")
    
    # Get dataloaders with optimized settings
    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True
    )
    
    # Train model
    model, _ = train_model(
        model_name=args.model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device
    )

if __name__ == "__main__":
    main()
