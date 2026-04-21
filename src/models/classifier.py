import torch.nn as nn
from torchvision.models import resnet18

class EventClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=101):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
