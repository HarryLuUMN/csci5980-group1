# model.py
"""
This module defines the model architectures for histopathology image classification.
It includes both Vision Transformer (ViT) and ResNet implementations, allowing for
easy switching between different model architectures.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16, resnet50, 
    ResNet50_Weights, ViT_B_16_Weights
)

from mambavision import create_model


class ViTForHistopathology(nn.Module):
    """
    Vision Transformer (ViT) model adapted for histopathology image classification.
    Uses a pretrained ViT-B/16 backbone with a custom classification head.
    """
    
    def __init__(self, num_classes=2):
        """
        Initialize the ViT model
        """
        super().__init__()
        # Load pretrained ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Get hidden dimension
        hidden_dim = self.vit.hidden_dim
        
        # Replace the classification head with a more regularized version
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.vit.heads = self.classifier

    def forward(self, x):
        """Forward pass"""
        return self.vit(x)

class ResNetForHistopathology(nn.Module):
    """
    ResNet model adapted for histopathology image classification.
    Uses a pretrained ResNet50 backbone with a custom classification head.
    """
    
    def __init__(self, num_classes=2):
        """
        Initialize the ResNet model
        """
        super().__init__()
        # Load pretrained ResNet
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer with a more regularized version
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        self.resnet.fc = self.classifier

    def forward(self, x):
        """Forward pass"""
        return self.resnet(x)

class MambaVisionModel(nn.Module):
    """
    MambaVision model adapted for specific tasks, with a customizable classification head.
    """
    
    def __init__(self, num_classes=1, model_path="/tmp/mambavision_tiny_1k.pth.tar"):
        """
        Initialize the MambaVision model with a custom classification head.
        
        Args:
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            model_path (str): Path to the pretrained model weights.
        """
        super().__init__()
        
        # Load the pretrained MambaVision model
        self.mamba_model = create_model('mamba_vision_T', pretrained=True, model_path=model_path)
        
        # Get the number of features from the existing head
        in_features = self.mamba_model.head.in_features
        
        # Replace the head with a custom classification head
        self.mamba_model.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.mamba_model(x)

def get_model(model_name='vit', num_classes=2):
    """
    Factory function to create and initialize a model
    """
    model_classes = {
        'vit': ViTForHistopathology,
        'resnet': ResNetForHistopathology,
        'mamba': MambaVisionModel
    }
    
    if model_name.lower() not in model_classes:
        raise ValueError(
            f"Model {model_name} not supported. "
            f"Choose from: {list(model_classes.keys())}"
        )
    
    return model_classes[model_name.lower()](num_classes=num_classes)

def save_checkpoint(model, save_path, epoch=None, optimizer=None, best_metric=None):
    """Save a model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model, checkpoint_path):
    """Load a model checkpoint"""
    model.load_state_dict(torch.load(checkpoint_path))
    return model
