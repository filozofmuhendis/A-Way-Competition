#!/usr/bin/env python3
"""
K2 Dataset Exoplanet Extraction Test Script
Bu script, feature_extracted__k2_dataset_clean.csv veri seti üzerinde
geliştirilmiş fuzzy logic ve PPO sistemi ile exoplanet extraction testi yapar.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from integrated_system import IntegratedExoplanetSystem
from fuzzy_logic_system import ExoplanetFuzzySystem
from enhanced_fuzzy_rules import EnhancedFuzzyRuleGenerator
from enhanced_ppo_agents import EnhancedMultiAgentPPOTrainer, FuzzyOutputType

def load_and_prepare_k2_data():
    """K2 veri setini yükle ve hazırla"""
    print("=" * 60)
    print("K2 VERİ SETİ YÜKLEME VE HAZIRLAMA")
    print("=" * 60)
    
    # Veri setini yükle
    df = pd.read_csv('feature_extracted__k2_dataset_clean.csv')
    print(f"✓ Veri seti yüklendi: {df.shape}")
    print(f"✓ Sütunlar: {len(df.columns)} adet")
    print(f"✓ Eksik değer: {df.isnull().sum().sum()} adet")
    
    # Label dağılımını göster
    print(f"\nLabel Dağılımı:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        label_name = "Exoplanet" if label == 1.0 else "Non-Exoplanet"
        print(f"  {label_name}: {count} ({percentage:.1f}%)")
    
    # Özellik sütunlarını ayır
    feature_columns = [col for col in df.columns if col not in ['id', 'label']]
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"\n✓ Özellik sayısı: {len(feature_columns)}")
    print(f"✓ Örnek sayısı: {len(X)}")
    
    # Veriyi normalize et
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Eğitim seti: {X_train.shape[0]} örnek")
    print(f"✓ Test seti: {X_test.shape[0]} örnek")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns,
        'scaler': scaler,
        'original_df': df
    }

def extract_fuzzy_features_from_k2(data_dict):
    """K2 veri setinden fuzzy logic için özellikler çıkar"""
    print("\n" + "=" * 60)
    print("FUZZY LOGIC ÖZELLİKLERİ ÇIKARMA")
    print("=" * 60)
    
    X = data_dict['X_train']
    feature_columns = data_dict['feature_columns']
    
    # K2 veri setindeki özelliklerden fuzzy özellikler türet
    fuzzy_features = []
    
    for i, sample in enumerate(X):
        # BLS (Box Least Squares) özelliklerini kullan
        bls_period_idx = feature_columns.index('bls_period') if 'bls_period' in feature_columns else -1
        bls_depth_idx = feature_columns.index('bls_depth') if 'bls_depth' in feature_columns else -1
        bls_duration_idx = feature_columns.index('bls_duration') if 'bls_duration' in feature_columns else -1
        
        # Flux özelliklerini kullan
        flux_std_idx = feature_columns.index('pdcsap_flux_std') if 'pdcsap_flux_std' in feature_columns else -1
        flux_mean_idx = feature_columns.index('pdcsap_flux_mean') if 'pdcsap_flux_mean' in feature_columns else -1
        
        # Fuzzy özellikler hesapla
        features = {
            'snr_transit': abs(sample[bls_depth_idx]) / (sample[flux_std_idx] + 1e-8) if bls_depth_idx >= 0 and flux_std_idx >= 0 else 0.5,
            'beta_factor': sample[bls_period_idx] / 10.0 if bls_period_idx >= 0 else 0.5,
            'depth_consistency': min(abs(sample[bls_depth_idx]) / 1000.0, 1.0) if bls_depth_idx >= 0 else 0.5,
            'duty_cycle': sample[bls_duration_idx] if bls_duration_idx >= 0 else 0.2,
            'odd_even_diff': abs(sample[flux_mean_idx] - np.median(X[:, flux_mean_idx])) / np.std(X[:, flux_mean_idx]) if flux_mean_idx >= 0 else 0.1
        }
        
        # Değerleri 0-1 arasına normalize et
        for key in features:
            features[key] = max(0, min(1, features[key]))
        
        fuzzy_features.append(features)
    
    print(f"✓ {len(fuzzy_features)} örnek için fuzzy özellikler çıkarıldı")
    print(f"✓ Özellik adları: {list(fuzzy_features[0].keys())}")
    
    return fuzzy_features

def test_fuzzy_analysis_on_k2(data_dict, fuzzy_features):
    """K2 veri seti üzerinde fuzzy analiz testi"""
    print("\n" + "=" * 60)
    print("FUZZY LOGIC ANALİZİ - K2 VERİ SETİ")
    print("=" * 60)
    
    # Fuzzy sistem oluştur
    fuzzy_system = ExoplanetFuzzySystem()
    
    # Fuzzy analiz sonuçları
    fuzzy_predictions = []
    fuzzy_probabilities = []
    
    # Fuzzy features'ı DataFrame'e çevir
    features_df = pd.DataFrame(fuzzy_features)
    
    try:
        # Fuzzy sistem predict metodunu kullan
        predictions = fuzzy_system.predict(features_df)
        
        # Predictions'ı 0-1 aralığına normalize et
        probabilities = predictions / 100.0  # 0-100 aralığından 0-1'e
        
        # Binary predictions oluştur
        binary_predictions = [1 if p > 0.5 else 0 for p in probabilities]
        
        fuzzy_predictions = binary_predictions
        fuzzy_probabilities = probabilities
        
    except Exception as e:
        print(f"⚠️  Fuzzy sistem hatası: {e}")
        # Fallback: rastgele tahminler
        fuzzy_predictions = [0] * len(fuzzy_features)
        fuzzy_probabilities = [0.0] * len(fuzzy_features)
    
    # Sonuçları değerlendir
    y_true = data_dict['y_train']
    accuracy = accuracy_score(y_true, fuzzy_predictions)
    
    print(f"✓ Fuzzy Logic Accuracy: {accuracy:.4f}")
    print(f"✓ Ortalama Exoplanet Probability: {np.mean(fuzzy_probabilities):.4f}")
    print(f"✓ Probability Std: {np.std(fuzzy_probabilities):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, fuzzy_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, fuzzy_predictions))
    
    return {
        'predictions': fuzzy_predictions,
        'probabilities': fuzzy_probabilities,
        'accuracy': accuracy
    }

def test_enhanced_system_on_k2(data_dict, fuzzy_features):
    """Geliştirilmiş sistem ile K2 testi"""
    print("\n" + "=" * 60)
    print("GELİŞTİRİLMİŞ SİSTEM TESTİ - K2 VERİ SETİ")
    print("=" * 60)
    
    # Geliştirilmiş entegre sistem oluştur
    enhanced_system = IntegratedExoplanetSystem(
        data_path='feature_extracted__k2_dataset_clean.csv',
        use_enhanced_system=True
    )
    
    print("✓ Geliştirilmiş sistem oluşturuldu")
    
    # Fuzzy analiz çalıştır
    print("\n--- Fuzzy Logic Analizi ---")
    fuzzy_results = []
    
    # Fuzzy features'ı DataFrame'e çevir
    features_df = pd.DataFrame(fuzzy_features[:10])  # İlk 10 örnek için test
    
    try:
        # Fuzzy sistem predict metodunu kullan
        predictions = enhanced_system.fuzzy_system.predict(features_df)
        probabilities = predictions / 100.0  # 0-1 aralığına normalize et
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            fuzzy_results.append({
                'sample_id': i,
                'probability': prob,
                'prediction': 1 if prob > 0.5 else 0,
                'true_label': data_dict['y_train'][i]
            })
            print(f"  Örnek {i}: P={prob:.4f}, Pred={1 if prob > 0.5 else 0}, True={int(data_dict['y_train'][i])}")
            
    except Exception as e:
        print(f"  ⚠️  Fuzzy analiz hatası: {e}")
        # Fallback sonuçlar
        for i in range(10):
            fuzzy_results.append({
                'sample_id': i,
                'probability': 0.0,
                'prediction': 0,
                'true_label': data_dict['y_train'][i]
            })
    
    # PPO eğitimi (kısa test)
    print("\n--- PPO Eğitim Testi ---")
    try:
        if enhanced_system.enhanced_ppo_trainer:
            print("✓ Özelleştirilmiş PPO trainer hazır")
            
            # Kısa eğitim testi (5 episode per combination)
            training_results = enhanced_system.enhanced_ppo_trainer.train_specialized_agents(
                episodes_per_combination=5
            )
            
            print("✓ PPO eğitim testi tamamlandı")
            print(f"✓ Eğitilen ajan sayısı: {len(training_results)}")
            
            # Sonuçları göster
            for agent_key, results in training_results.items():
                print(f"  {agent_key}: Final Reward = {results['final_avg_reward']:.4f}")
        
    except Exception as e:
        print(f"⚠️  PPO eğitim hatası: {e}")
    
    return {
        'fuzzy_results': fuzzy_results,
        'system_stats': enhanced_system.get_enhanced_system_stats()
    }

def visualize_k2_results(data_dict, fuzzy_results, enhanced_results):
    """K2 sonuçlarını görselleştir"""
    print("\n" + "=" * 60)
    print("SONUÇLARIN GÖRSELLEŞTİRİLMESİ")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K2 Dataset Exoplanet Extraction Results', fontsize=16)
    
    # 1. Label dağılımı
    df = data_dict['original_df']
    label_counts = df['label'].value_counts()
    axes[0, 0].pie(label_counts.values, labels=['Non-Exoplanet', 'Exoplanet'], autopct='%1.1f%%')
    axes[0, 0].set_title('K2 Dataset Label Distribution')
    
    # 2. Fuzzy probability dağılımı
    if 'probabilities' in fuzzy_results:
        axes[0, 1].hist(fuzzy_results['probabilities'], bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Fuzzy Logic Probability Distribution')
        axes[0, 1].set_xlabel('Exoplanet Probability')
        axes[0, 1].set_ylabel('Frequency')
    
    # 3. Özellik korelasyonu (ilk 10 özellik)
    feature_cols = data_dict['feature_columns'][:10]
    corr_data = df[feature_cols].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix (Top 10)')
    
    # 4. Accuracy karşılaştırması
    methods = ['Fuzzy Logic']
    accuracies = [fuzzy_results.get('accuracy', 0)]
    
    axes[1, 1].bar(methods, accuracies, color=['blue'])
    axes[1, 1].set_title('Method Comparison - Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    
    # Accuracy değerlerini göster
    for i, acc in enumerate(accuracies):
        axes[1, 1].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('k2_exoplanet_extraction_results.png', dpi=300, bbox_inches='tight')
    print("✓ Sonuçlar 'k2_exoplanet_extraction_results.png' dosyasına kaydedildi")
    
    return fig

def run_k2_exoplanet_extraction_test():
    """Ana test fonksiyonu"""
    print("🚀 K2 DATASET EXOPLANET EXTRACTION TEST BAŞLADI")
    print("=" * 80)
    
    try:
        # 1. Veri hazırlama
        data_dict = load_and_prepare_k2_data()
        
        # 2. Fuzzy özellikler çıkarma
        fuzzy_features = extract_fuzzy_features_from_k2(data_dict)
        
        # 3. Fuzzy analiz testi
        fuzzy_results = test_fuzzy_analysis_on_k2(data_dict, fuzzy_features)
        
        # 4. Geliştirilmiş sistem testi
        enhanced_results = test_enhanced_system_on_k2(data_dict, fuzzy_features)
        
        # 5. Sonuçları görselleştir
        visualize_k2_results(data_dict, fuzzy_results, enhanced_results)
        
        # 6. Özet rapor
        print("\n" + "=" * 80)
        print("TEST ÖZET RAPORU")
        print("=" * 80)
        print(f"✓ Toplam örnek sayısı: {len(data_dict['y_train'])}")
        print(f"✓ Özellik sayısı: {len(data_dict['feature_columns'])}")
        print(f"✓ Fuzzy Logic Accuracy: {fuzzy_results['accuracy']:.4f}")
        print(f"✓ Ortalama Exoplanet Probability: {np.mean(fuzzy_results['probabilities']):.4f}")
        
        if enhanced_results['system_stats']:
            stats = enhanced_results['system_stats']
            print(f"✓ Özelleştirilmiş PPO Ajanları: {stats['total_specialized_agents']} adet")
            print(f"✓ Fuzzy Çıktı Tipleri: {stats['fuzzy_output_types']} adet")
        
        print("\n🎉 K2 DATASET EXOPLANET EXTRACTION TEST TAMAMLANDI!")
        
        return {
            'data': data_dict,
            'fuzzy_results': fuzzy_results,
            'enhanced_results': enhanced_results
        }
        
    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_k2_exoplanet_extraction_test()