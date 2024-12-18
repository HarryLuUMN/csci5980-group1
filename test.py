# test.py
"""
Test script for evaluating trained histopathology classification models.
Supports both ViT and ResNet architectures and saves results in model-specific directories.
"""

import os
import torch
import torch.nn as nn
import json
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                           recall_score, f1_score, roc_curve)

from model import get_model
from dataloader import get_dataloaders
from metricstracker import MetricsTracker

def test_model(model_name, test_loader, device='cuda'):
    """
    Evaluate a trained model on the test set
    
    Args:
        model_name (str): Name of the model architecture to use
        test_loader (DataLoader): DataLoader for test data
        device (str): Device to test on ('cuda' or 'cpu')
        
    Returns:
        dict: Dictionary containing test metrics
    """
    # Load model
    model = get_model(model_name, num_classes=2)
    model_path = f"run/{model_name}/models/{model_name}_best.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Initialize metrics
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_preds = []
    test_probs = []
    test_labels = []
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Testing {model_name.upper()}'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)
            
            test_probs.extend(probabilities)
            test_preds.extend(predictions)
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'test_loss': test_loss / len(test_loader),
        'test_auc': roc_auc_score(test_labels, test_probs),
        'test_acc': accuracy_score(test_labels, test_preds),
        'test_precision': precision_score(test_labels, test_preds),
        'test_recall': recall_score(test_labels, test_preds),
        'test_f1': f1_score(test_labels, test_preds)
    }
    
    # Create metrics tracker for visualization
    output_dirs = {
        'base': 'run',
        'models': 'run/models',
        'metrics': 'run/metrics',
        'plots': 'run/plots'
    }
    
    metrics_tracker = MetricsTracker(output_dirs)
    
    # Generate and save plots
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    metrics_tracker.plot_roc_curve(fpr, tpr, metrics['test_auc'], phase='test')
    metrics_tracker.plot_confusion_matrix(test_labels, test_preds, phase='test')
    
    # Save test results
    save_test_results(metrics, model_name)
    
    return metrics

def save_test_results(metrics, model_name):
    """
    Save test results to JSON and CSV files
    
    Args:
        metrics (dict): Dictionary containing test metrics
        model_name (str): Name of the model architecture
    """
    # Create output directory if it doesn't exist
    metrics_dir = f"run/{model_name}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(metrics_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save as CSV
    csv_path = os.path.join(metrics_dir, 'test_results.csv')
    df = pd.DataFrame([metrics])
    df.to_csv(csv_path, index=False)
    
    print(f"\nTest results saved to:")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Test histopathology classification model')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet'],
                      help='Model architecture to use (vit or resnet)')
    parser.add_argument('--data_root', type=str, default='data',
                      help='Root directory for the dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=32,
                      help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get test dataloader
    _, _, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Test model
    metrics = test_model(args.model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Loss: {metrics['test_loss']:.4f}")
    print(f"AUC: {metrics['test_auc']:.4f}")
    print(f"Accuracy: {metrics['test_acc']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall: {metrics['test_recall']:.4f}")
    print(f"F1 Score: {metrics['test_f1']:.4f}")

if __name__ == "__main__":
    main()