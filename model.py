import torch.nn as nn
import torchvision.models as models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
