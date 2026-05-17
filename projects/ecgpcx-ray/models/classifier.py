"""Classes for baselines classifier for binary chest X-ray classification (Pneumonia vs Healthy)."""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """4-block CNN for binary chest X-ray classification.

    Input:  (B, 1, H, W) grayscale image, normalized to [0, 1]
    Output: (B, 1) raw logit for the pneumonia class
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # → (B, 32,  H/2,  W/2)
            ConvBlock(32, 64),  # → (B, 64,  H/4,  W/4)
            ConvBlock(64, 128),  # → (B, 128, H/8,  W/8)
            ConvBlock(128, 256), # → (B, 256, H/16, W/16)
            ConvBlock(256, 512), # → (B, 512, H/32, W/32)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class FrozenResNet18(nn.Module):
    """ResNet-18 com backbone ImageNet congelado e head binário.

    Aceita imagens grayscale (B, 1, H, W): repete o canal 3x internamente
    e aplica a normalização ImageNet antes de passar pelo backbone.
    Somente a camada fc é treinável.
    """

    # from ImageNet (used to normalize inputs)
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze the entire backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Replace the final classifier with a binary output
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)

        self.model = backbone

        mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self._STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
        x = (x - self.mean) / self.std  # ImageNet normalization
        return self.model(x)  # (B, 1)
