
import torch
import torch.nn as nn
import torch.nn.functional as F

class IntelCNNPyTorch(nn.Module):
    def __init__(self):
        super(IntelCNNPyTorch, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 6)  # 6 classes
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (32, 75, 75)
        x = self.pool(F.relu(self.conv2(x)))   # (64, 37, 37)
        x = self.pool(F.relu(self.conv3(x)))   # (128, 18, 18)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
