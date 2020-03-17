from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(100, 10))

    def forward(self, x):
        feature_vector = self.feature_extractor(x)
        labels = self.classifier(feature_vector)
        return feature_vector, labels

