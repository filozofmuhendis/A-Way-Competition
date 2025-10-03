import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class ExoplanetCNN(nn.Module):
    def __init__(self, input_length=3197):
        super(ExoplanetCNN, self).__init__()
        
        # 1D Convolutional layers for time series
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 100, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ExoplanetTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 99.0]).to(device))  # Class imbalance
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
        return total_loss / len(train_loader), predictions, targets
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                prob = F.softmax(output, dim=1)[:, 1]  # Probability of positive class
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                probabilities.extend(prob.cpu().numpy())
        
        return total_loss / len(test_loader), predictions, targets, probabilities
    
    def calculate_metrics(self, predictions, targets, probabilities):
        # Convert labels: 2->1 (exoplanet), 1->0 (no exoplanet)
        targets_binary = [1 if t == 2 else 0 for t in targets]
        
        metrics = {
            'accuracy': accuracy_score(targets_binary, predictions),
            'precision': precision_score(targets_binary, predictions, zero_division=0),
            'recall': recall_score(targets_binary, predictions, zero_division=0),
            'f1_score': f1_score(targets_binary, predictions, zero_division=0),
            'roc_auc': roc_auc_score(targets_binary, probabilities) if len(set(targets_binary)) > 1 else 0
        }
        
        return metrics