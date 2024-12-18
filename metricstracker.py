# metricstracker.py
"""
This module provides functionality for tracking, saving, and visualizing various metrics
during model training and evaluation. It handles both performance metrics and visualization
of results through plots and charts.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MetricsTracker:
    """
    A class to track, save, and visualize various metrics during model training and evaluation.
    
    This class handles:
    - Tracking metrics over epochs
    - Saving metrics to CSV and JSON
    - Plotting learning curves
    - Generating confusion matrices
    - Creating ROC curves
    """
    
    def __init__(self, output_dirs):
        """
        Initialize the MetricsTracker with output directories for saving results
        
        Args:
            output_dirs (dict): Dictionary containing paths for saving different outputs
                             (metrics, plots, etc.)
        """
        self.output_dirs = output_dirs
        self.history = {
            'epoch': [],
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        
    def update(self, epoch, metrics):
        """
        Update the metrics history with new values
        
        Args:
            epoch (int): Current epoch number
            metrics (dict): Dictionary containing metric names and their values
        """
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_metrics(self):
        """Save the metrics history to both CSV and JSON formats"""
        # Save as CSV for easy viewing and analysis
        df = pd.DataFrame(self.history)
        df.to_csv(f"{self.output_dirs['metrics']}/training_history.csv", index=False)
        
        # Save as JSON for programmatic access
        with open(f"{self.output_dirs['metrics']}/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_metrics(self):
        """Generate plots for all tracked metrics"""
        metrics_to_plot = [
            ('loss', 'Model Loss'),
            ('auc', 'Model AUC'),
            ('acc', 'Model Accuracy'),
            ('precision', 'Model Precision'),
            ('recall', 'Model Recall')
        ]
        
        for metric_name, title in metrics_to_plot:
            self._plot_metric(metric_name, title)
        
        plt.close('all')
    
    def _plot_metric(self, metric_name, title):
        """
        Create a plot for a specific metric
        
        Args:
            metric_name (str): Name of the metric to plot
            title (str): Title for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.history['epoch'], 
            self.history[f'train_{metric_name}'], 
            label=f'Train {metric_name}'
        )
        plt.plot(
            self.history['epoch'], 
            self.history[f'val_{metric_name}'], 
            label=f'Val {metric_name}'
        )
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dirs['plots']}/{metric_name}_curves.png")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, phase='val'):
        """
        Generate and save a confusion matrix plot
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            phase (str): Phase identifier ('train', 'val', or 'test')
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {phase.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{self.output_dirs['plots']}/confusion_matrix_{phase}.png")
        plt.close()
    
    def plot_roc_curve(self, fpr, tpr, auc, phase='val'):
        """
        Generate and save an ROC curve plot
        
        Args:
            fpr (array-like): False positive rates
            tpr (array-like): True positive rates
            auc (float): Area under the curve value
            phase (str): Phase identifier ('train', 'val', or 'test')
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {phase.capitalize()} Set')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_dirs['plots']}/roc_curve_{phase}.png")
        plt.close()