import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ---------- SNN Yardımcı Fonksiyonlar ----------
def detrend_and_znorm(x, win=101):
    """Trend kaldırma ve z-normalizasyon"""
    if len(x) < win:
        win = len(x) // 2 if len(x) > 2 else 1
    
    kernel = np.ones(win) / win
    trend = np.convolve(x, kernel, mode="same")
    detr = x - trend + np.mean(trend)
    std = np.std(detr)
    if std < 1e-12: 
        std = 1.0
    return (detr - np.median(detr)) / std

def encode_spikes(z, thr=-0.8, deriv_q=0.90):
    """Spike kodlama: eşik ve türev tabanlı"""
    s_thr = (z < thr).astype(np.int8)
    dz = np.diff(z, prepend=z[0])
    gamma = np.quantile(np.abs(dz), deriv_q)
    s_der = (dz < -gamma).astype(np.int8)
    return s_thr, s_der

def lif_layer(stream, leak=0.02, thresholds=(0.9, 1.8, 2.7), refrac=25):
    """Leaky Integrate-and-Fire nöron katmanı"""
    V = np.zeros(len(thresholds), dtype=float)
    refr = np.zeros(len(thresholds), dtype=int)
    outs = np.zeros((len(thresholds), len(stream)), dtype=np.int8)
    
    for i, s in enumerate(stream):
        active = refr == 0
        V[active] = (1 - leak) * V[active] + s
        fired = (V >= thresholds) & active
        outs[fired, i] = 1
        V[fired] = 0.0
        refr[fired] = refrac
        refr[~active] -= 1
    
    return outs

def extract_snn_features(flux_series, detrend_win=101, thr=-0.8, deriv_q=0.90):
    """Tek bir flux serisi için SNN öznitelik çıkarımı"""
    # Z-normalizasyon ve trend kaldırma
    z = detrend_and_znorm(flux_series.astype(float), win=detrend_win)
    
    # Spike kodlama
    s_thr, s_der = encode_spikes(z, thr=thr, deriv_q=deriv_q)
    
    # LIF toplulaştırma
    current = 0.8 * s_thr + 0.5 * s_der
    lif = lif_layer(current)
    
    # Öznitelik çıkarımı
    conv_win = min(64, len(flux_series) // 4)
    conv = lambda s: float(np.convolve(s, np.ones(conv_win), mode="same").max()) if len(s) > 0 else 0.0
    
    features = {
        # Temel istatistikler
        "mean": float(np.mean(z)),
        "std": float(np.std(z)),
        "min": float(np.min(z)),
        "max": float(np.max(z)),
        "skewness": float(np.mean((z - np.mean(z))**3) / (np.std(z)**3 + 1e-8)),
        "kurtosis": float(np.mean((z - np.mean(z))**4) / (np.std(z)**4 + 1e-8)),
        "slope": float(np.polyfit(np.arange(len(z)), z, 1)[0]),
        
        # Spike öznitelikleri
        "thr_total": float(s_thr.sum()),
        "thr_peak": conv(s_thr),
        "thr_density": float(s_thr.sum() / len(s_thr)),
        "der_total": float(s_der.sum()),
        "der_peak": conv(s_der),
        "der_density": float(s_der.sum() / len(s_der)),
        
        # LIF nöron öznitelikleri
        "lif1_total": float(lif[0].sum()),
        "lif1_peak": conv(lif[0]),
        "lif1_density": float(lif[0].sum() / len(lif[0])),
        "lif2_total": float(lif[1].sum()),
        "lif2_peak": conv(lif[1]),
        "lif2_density": float(lif[1].sum() / len(lif[1])),
        "lif3_total": float(lif[2].sum()),
        "lif3_peak": conv(lif[2]),
        "lif3_density": float(lif[2].sum() / len(lif[2])),
        
        # Ek öznitelikler
        "energy": float(np.sum(z**2)),
        "zero_crossings": float(np.sum(np.diff(np.sign(z)) != 0)),
        "peak_to_peak": float(np.max(z) - np.min(z))
    }
    
    return features

class SNNExoplanetDetector:
    """SNN tabanlı Exoplanet Dedektörü"""
    
    def __init__(self, contamination='auto', n_estimators=300, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.detector = None
        self.feature_names = None
        
    def extract_features_from_dataset(self, df):
        """Dataset'ten tüm örnekler için öznitelik çıkarımı"""
        print("SNN öznitelik çıkarımı başlıyor...")
        
        # FLUX sütunlarını al - farklı veri setleri için uyumlu
        if 'FLUX.1' in df.columns:
            flux_columns = [col for col in df.columns if col.startswith('FLUX.')]
            label_col = 'LABEL'
        elif 'flux_0' in df.columns:
            flux_columns = [col for col in df.columns if col.startswith('flux_')]
            label_col = 'label'
        elif 'pdcsap_flux' in df.columns:
            flux_columns = ['pdcsap_flux']
            label_col = 'label'
        else:
            raise ValueError("Desteklenen flux sütunları bulunamadı.")
        
        labels = df[label_col].values if label_col in df.columns else None
        
        features_list = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"İşlenen örnek: {idx}/{len(df)}")
            
            if len(flux_columns) == 1:
                # K2 dataset için tek sütun
                flux_series = np.array([row[flux_columns[0]]])
            else:
                # exoTrain ve koi_v1 için çoklu sütun
                flux_series = row[flux_columns].values
                
            # NaN değerleri temizle
            flux_series = flux_series[~np.isnan(flux_series)]
            
            if len(flux_series) > 10:  # Minimum veri uzunluğu kontrolü
                features = extract_snn_features(flux_series)
                features_list.append(features)
            else:
                # Boş öznitelikler ekle
                features_list.append({key: 0.0 for key in [
                    "mean", "std", "min", "max", "skewness", "kurtosis", "slope",
                    "thr_total", "thr_peak", "thr_density", "der_total", "der_peak", "der_density",
                    "lif1_total", "lif1_peak", "lif1_density", "lif2_total", "lif2_peak", "lif2_density",
                    "lif3_total", "lif3_peak", "lif3_density", "energy", "zero_crossings", "peak_to_peak"
                ]})
        
        features_df = pd.DataFrame(features_list)
        
        if labels is not None:
            features_df[label_col] = labels
            
        self.feature_names = [col for col in features_df.columns if col not in ['LABEL', 'label']]
        print(f"Öznitelik çıkarımı tamamlandı. Toplam öznitelik sayısı: {len(self.feature_names)}")
        
        return features_df
    
    def fit(self, features_df):
        """Modeli eğit"""
        X = features_df[self.feature_names].values
        
        # Öznitelikleri normalize et
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest ile anomali tespiti
        self.detector = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Isolation Forest eğitimi başlıyor...")
        self.detector.fit(X_scaled)
        print("Model eğitimi tamamlandı.")
        
        return self
    
    def predict(self, features_df):
        """Anomali skorları ve tahminler"""
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        # Anomali skorları (-1: anomali, 1: normal)
        predictions = self.detector.predict(X_scaled)
        # Anomali skorları (büyük değer = daha anomal)
        scores = -self.detector.score_samples(X_scaled)
        
        return predictions, scores
    
    def detect_exoplanets(self, features_df, threshold_percentile=95):
        """Exoplanet detection using anomaly scores"""
        # Check for label column (either 'LABEL' or 'label')
        if 'LABEL' in features_df.columns:
            label_col = 'LABEL'
            # For exoTrain: 2 = planet, 1 = no planet
            true_labels = (features_df['LABEL'] == 2).astype(int)
        elif 'label' in features_df.columns:
            label_col = 'label'
            # For koi_v1 and k2: 1 = planet, 0 = no planet
            true_labels = features_df['label'].astype(int)
        else:
            print("Değerlendirme için LABEL veya label sütunu gerekli!")
            return None
        
        predictions, scores = self.predict(features_df)
        
        # Eşik belirleme
        threshold = np.percentile(scores, threshold_percentile)
        binary_predictions = (scores >= threshold).astype(int)
        
        # Metrikleri hesapla
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, binary_predictions, average='binary')
        cm = confusion_matrix(true_labels, binary_predictions)
        roc_auc = roc_auc_score(true_labels, scores)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'scores': scores,
            'predictions': binary_predictions,
            'true_labels': true_labels
        }
        
        return results
    
    def plot_results(self, results, features_df):
        """Sonuçları görselleştir"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Anomali skorları dağılımı
        axes[0, 0].hist(results['scores'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(results['threshold'], color='red', linestyle='--', 
                          label=f'Eşik: {results["threshold"]:.3f}')
        axes[0, 0].set_xlabel('Anomali Skoru')
        axes[0, 0].set_ylabel('Frekans')
        axes[0, 0].set_title('Anomali Skorları Dağılımı')
        axes[0, 0].legend()
        
        # 2. Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_xlabel('Tahmin Edilen')
        axes[0, 1].set_ylabel('Gerçek')
        axes[0, 1].set_title('Confusion Matrix')
        
        # 3. Sınıflara göre skor dağılımı
        planet_scores = results['scores'][results['true_labels'] == 1]
        no_planet_scores = results['scores'][results['true_labels'] == 0]
        
        axes[1, 0].hist(no_planet_scores, bins=30, alpha=0.7, label='Planet Yok', color='blue')
        axes[1, 0].hist(planet_scores, bins=30, alpha=0.7, label='Planet Var', color='red')
        axes[1, 0].axvline(results['threshold'], color='black', linestyle='--', label='Eşik')
        axes[1, 0].set_xlabel('Anomali Skoru')
        axes[1, 0].set_ylabel('Frekans')
        axes[1, 0].set_title('Sınıflara Göre Skor Dağılımı')
        axes[1, 0].legend()
        
        # 4. Öznitelik önem sıralaması (basit analiz)
        feature_importance = np.abs(np.corrcoef(features_df[self.feature_names].T, 
                                               results['scores'])[:-1, -1])
        top_features = np.argsort(feature_importance)[-10:]
        
        axes[1, 1].barh(range(len(top_features)), feature_importance[top_features])
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels([self.feature_names[i] for i in top_features])
        axes[1, 1].set_xlabel('Korelasyon (Mutlak Değer)')
        axes[1, 1].set_title('En Önemli 10 Öznitelik')
        
        plt.tight_layout()
        plt.show()
        
        # Performans metrikleri yazdır
        print("\n" + "="*50)
        print("SNN-ML EXOPLANET DETECTOR SONUÇLARI")
        print("="*50)
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Precision (Planet Detection): {results['precision']:.4f}")
        print(f"Recall (Planet Detection): {results['recall']:.4f}")
        print(f"Eşik Değeri: {results['threshold']:.4f}")
        print(f"Tespit Edilen Planet Sayısı: {np.sum(results['predictions'])}")
        print(f"Gerçek Planet Sayısı: {np.sum(results['true_labels'])}")

def main():
    """Ana fonksiyon"""
    print("SNN-ML Exoplanet Detector başlatılıyor...")
    
    # Veri yükle - koi_v1.csv'yi öncelikle kontrol et
    print("Veri yükleniyor...")
    if Path("koi_v1.csv").exists():
        df = pd.read_csv("koi_v1.csv")
        label_col = "label"
        output_file = "snn_exoplanet_features_koi_v1.csv"
    elif Path("exoTrain.csv").exists():
        df = pd.read_csv("exoTrain.csv")
        label_col = "LABEL"
        output_file = "snn_exoplanet_features_exoTrain.csv"
    elif Path("k2_dataset.csv").exists():
        df = pd.read_csv("k2_dataset.csv")
        label_col = "label"
        output_file = "snn_exoplanet_features_k2.csv"
    else:
        raise FileNotFoundError("Veri dosyası bulunamadı. koi_v1.csv, exoTrain.csv veya k2_dataset.csv dosyalarından birini ekleyin.")
    
    print(f"Yüklenen veri boyutu: {df.shape}")
    print(f"Sınıf dağılımı:\n{df[label_col].value_counts()}")
    
    # Detector oluştur
    detector = SNNExoplanetDetector(contamination=0.05, n_estimators=300, random_state=42)
    
    # Öznitelik çıkarımı
    features_df = detector.extract_features_from_dataset(df)
    
    # Öznitelikleri kaydet
    features_df.to_csv(output_file, index=False)
    print(f"Öznitelikler '{output_file}' dosyasına kaydedildi.")
    
    # Cross-validation ile değerlendirme
    print("\n5-Fold Cross Validation başlıyor...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    # Label sütununu standardize et
    if label_col == "LABEL":
        y = (features_df['LABEL'] == 2).astype(int)  # exoTrain için
    else:
        y = features_df['label'].astype(int)  # koi_v1 ve k2_dataset için
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features_df, y)):
        print(f"Fold {fold + 1}/5")
        
        train_features = features_df.iloc[train_idx]
        val_features = features_df.iloc[val_idx]
        
        # Model eğit
        fold_detector = SNNExoplanetDetector(contamination=0.05, n_estimators=300, random_state=42)
        fold_detector.feature_names = detector.feature_names
        fold_detector.fit(train_features)
        
        # Değerlendir
        results = fold_detector.detect_exoplanets(val_features)
        cv_scores.append(results['f1_score'])
        
        print(f"Fold {fold + 1} F1-Score: {results['f1_score']:.4f}")
    
    print(f"\nOrtalama CV F1-Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Tüm veri ile final model
    print("\nFinal model eğitiliyor...")
    detector.fit(features_df)
    final_results = detector.detect_exoplanets(features_df)
    
    # Sonuçları görselleştir
    detector.plot_results(final_results, features_df)
    
    return detector, features_df, final_results

if __name__ == "__main__":
    detector, features_df, results = main()