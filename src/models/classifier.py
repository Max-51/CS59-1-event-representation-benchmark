import torch.nn as nn
from torchvision.models import resnet18


class EventClassifier(nn.Module):
    """ResNet18 backbone with configurable input channels and num_classes."""

    def __init__(self, in_channels: int, num_classes: int = 101):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Linear(512, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
