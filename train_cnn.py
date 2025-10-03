import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cnn_model import ExoplanetCNN, ExoplanetTrainer
import matplotlib.pyplot as plt

class ExoplanetDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_data():
    # Veri yükleme
    train_df = pd.read_csv('exoTrain.csv')
    test_df = pd.read_csv('exoTest.csv')
    
    # Özellik ve etiket ayrımı
    X_train = train_df.drop('LABEL', axis=1).values
    y_train = train_df['LABEL'].values
    X_test = test_df.drop('LABEL', axis=1).values
    y_test = test_df['LABEL'].values
    
    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Etiket dönüşümü: 2->1, 1->0
    y_train = np.where(y_train == 2, 1, 0)
    y_test = np.where(y_test == 2, 1, 0)
    
    return X_train, X_test, y_train, y_test

def create_weighted_sampler(labels):
    """Dengesiz veri seti için ağırlıklı örnekleme"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_model():
    # Veri hazırlama
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Dataset ve DataLoader oluşturma
    train_dataset = ExoplanetDataset(X_train, y_train)
    test_dataset = ExoplanetDataset(X_test, y_test)
    
    # Weighted sampler for imbalanced data
    sampler = create_weighted_sampler(y_train)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model ve trainer oluşturma
    model = ExoplanetCNN(input_length=X_train.shape[1])
    trainer = ExoplanetTrainer(model)
    
    # Eğitim döngüsü
    num_epochs = 100
    train_losses = []
    test_losses = []
    best_f1 = 0
    
    print("Eğitim başlıyor...")
    print(f"Toplam eğitim örneği: {len(X_train)}")
    print(f"Toplam test örneği: {len(X_test)}")
    print(f"Pozitif örnek oranı: {np.mean(y_train):.4f}")
    
    for epoch in range(num_epochs):
        # Eğitim
        train_loss, train_preds, train_targets = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Test
        test_loss, test_preds, test_targets, test_probs = trainer.evaluate(test_loader)
        test_losses.append(test_loss)
        
        # Metrikleri hesapla
        train_metrics = trainer.calculate_metrics(train_preds, train_targets, [])
        test_metrics = trainer.calculate_metrics(test_preds, test_targets, test_probs)
        
        # Learning rate scheduler
        trainer.scheduler.step(test_loss)
        
        # En iyi modeli kaydet
        if test_metrics['f1_score'] > best_f1:
            best_f1 = test_metrics['f1_score']
            torch.save(model.state_dict(), 'best_exoplanet_cnn.pth')
        
        # Her 10 epoch'ta sonuçları yazdır
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"Test Metrics:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    return model, train_losses, test_losses, test_metrics

if __name__ == "__main__":
    model, train_losses, test_losses, final_metrics = train_model()
    
    # Sonuçları görselleştir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    metrics_names = list(final_metrics.keys())
    metrics_values = list(final_metrics.values())
    plt.bar(metrics_names, metrics_values)
    plt.ylabel('Score')
    plt.title('Final Test Metrics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('cnn_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== FINAL RESULTS ===")
    print(f"Best Test Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")