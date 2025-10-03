import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_models import *
from train_cnn import ExoplanetDataset, prepare_data, create_weighted_sampler
from cnn_model import ExoplanetCNN
import warnings
warnings.filterwarnings('ignore')

class AdvancedTrainer:
    def __init__(self, model, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 99.0]).to(device)
        )
        
        # Optimizer with different learning rates for different models
        if 'Transformer' in model_name:
            self.optimizer = torch.optim.AdamW(model .parameters(), lr=0.0001, weight_decay=1e-4)
        elif 'WaveNet' in model_name:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
    
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        return total_loss / len(train_loader), predictions, targets
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                prob = F.softmax(output, dim=1)[:, 1]
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
                probabilities.extend(prob.cpu().numpy())
        
        return total_loss / len(val_loader), predictions, targets, probabilities
    
    def calculate_metrics(self, predictions, targets, probabilities):
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1_score': f1_score(targets, predictions, zero_division=0),
            'roc_auc': roc_auc_score(targets, probabilities) if len(set(targets)) > 1 else 0
        }
        return metrics

def train_all_models():
    # Veri hazırlama
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Model tanımları
    models = {
        'CNN_Basic': ExoplanetCNN(X_train.shape[1]),
        'ResNet1D': ResNet1D(X_train.shape[1]),
        'Transformer': TransformerEncoder(X_train.shape[1]),
        'LSTM_Attention': LSTMAttention(),
        'WaveNet': WaveNet(),
        'EfficientNet1D': EfficientNet1D(X_train.shape[1])
    }
    
    results = {}
    
    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_class in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nFold {fold + 1}/5")
            
            # Fold veri ayırma
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Dataset ve DataLoader
            train_dataset = ExoplanetDataset(X_fold_train, y_fold_train)
            val_dataset = ExoplanetDataset(X_fold_val, y_fold_val)
            
            sampler = create_weighted_sampler(y_fold_train)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Model ve trainer
            model = model_class
            trainer = AdvancedTrainer(model, model_name)
            
            best_f1 = 0
            patience = 0
            max_patience = 15
            
            # Eğitim döngüsü
            for epoch in range(50):
                train_loss, train_preds, train_targets = trainer.train_epoch(train_loader)
                val_loss, val_preds, val_targets, val_probs = trainer.validate(val_loader)
                
                metrics = trainer.calculate_metrics(val_preds, val_targets, val_probs)
                trainer.scheduler.step()
                
                # Early stopping
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    patience = 0
                    # Save best model
                    torch.save(model.state_dict(), f'best_{model_name}_fold{fold}.pth')
                else:
                    patience += 1
                
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Val F1={metrics['f1_score']:.4f}, Val AUC={metrics['roc_auc']:.4f}")
            
            fold_results.append({
                'fold': fold,
                'best_f1': best_f1,
                'final_metrics': metrics
            })
        
        # Fold sonuçlarını ortala
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            values = [fold['final_metrics'][metric] for fold in fold_results]
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        results[model_name] = avg_metrics
        
        print(f"\n{model_name} Results (5-Fold CV):")
        for metric, stats in avg_metrics.items():
            print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return results

def visualize_results(results):
    # Sonuçları DataFrame'e çevir
    data = []
    for model, metrics in results.items():
        for metric, stats in metrics.items():
            data.append({
                'Model': model,
                'Metric': metric.upper(),
                'Mean': stats['mean'],
                'Std': stats['std']
            })
    
    df = pd.DataFrame(data)
    
    # Pivot table oluştur
    pivot_mean = df.pivot(index='Model', columns='Metric', values='Mean')
    pivot_std = df.pivot(index='Model', columns='Metric', values='Std')
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Heatmap - Mean values
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('Mean Performance Metrics')
    
    # Bar plot - F1 Score
    f1_data = df[df['Metric'] == 'F1_SCORE']
    axes[0,1].bar(f1_data['Model'], f1_data['Mean'], yerr=f1_data['Std'], capsize=5)
    axes[0,1].set_title('F1-Score Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Bar plot - ROC AUC
    auc_data = df[df['Metric'] == 'ROC_AUC']
    axes[1,0].bar(auc_data['Model'], auc_data['Mean'], yerr=auc_data['Std'], capsize=5, color='green')
    axes[1,0].set_title('ROC-AUC Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall scatter
    prec_data = df[df['Metric'] == 'PRECISION']
    rec_data = df[df['Metric'] == 'RECALL']
    
    for i, model in enumerate(prec_data['Model']):
        prec = prec_data.iloc[i]['Mean']
        rec = rec_data.iloc[i]['Mean']
        axes[1,1].scatter(prec, rec, s=100, label=model)
    
    axes[1,1].set_xlabel('Precision')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].set_title('Precision vs Recall')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('advanced_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_mean

if __name__ == "__main__":
    print("Starting advanced model training...")
    results = train_all_models()
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    summary_df = visualize_results(results)
    print(summary_df.round(4))
    
    # En iyi modeli belirle
    best_model = summary_df['F1_SCORE'].idxmax()
    print(f"\nBest performing model: {best_model}")
    print(f"F1-Score: {summary_df.loc[best_model, 'F1_SCORE']:.4f}")
    print(f"ROC-AUC: {summary_df.loc[best_model, 'ROC_AUC']:.4f}")