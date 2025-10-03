import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AudioInspiredPreprocessor:
    """
    Audio-inspired preprocessing for exoplanet detection
    Adapted for K2 dataset with fuzzy logic and SVM neural network integration
    """
    
    def __init__(self, win_len=256, hop=64, noise_pctl=20, gate_beta=1.2, 
                 pre_emph_alpha=0.97, roll_med_win=7):
        # STFT parametreleri
        self.win_len = win_len
        self.hop = hop
        self.noise_pctl = noise_pctl
        self.gate_beta = gate_beta
        
        # Pre-emphasis parametreleri
        self.pre_emph_alpha = pre_emph_alpha
        self.roll_med_win = roll_med_win
        
        # SNN parametreleri
        self.spike_thr = -0.8
        self.deriv_q = 0.90
        self.lif_leak = 0.02
        self.lif_thresholds = (0.9, 1.8, 2.7)
        self.lif_refrac = 25
        
        # Scaler
        self.scaler = RobustScaler()
        
    def pre_emphasis(self, x, alpha=None):
        """Pre-emphasis filter"""
        if alpha is None:
            alpha = self.pre_emph_alpha
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - alpha * x[:-1]
        return y
    
    def rolling_median(self, x, win=None):
        """Rolling median filter"""
        if win is None:
            win = self.roll_med_win
        return pd.Series(x).rolling(window=win, center=True, min_periods=1).median().to_numpy()
    
    def stft_1d(self, sig, win_len=None, hop=None):
        """1D Short-Time Fourier Transform"""
        if win_len is None:
            win_len = self.win_len
        if hop is None:
            hop = self.hop
            
        n = len(sig)
        win = np.hanning(win_len)
        frames, centers = [], []
        
        for start in range(0, n - win_len + 1, hop):
            frames.append(np.fft.rfft(sig[start:start+win_len] * win))
            centers.append(start + win_len//2)
        
        return np.vstack(frames), np.array(centers), win
    
    def istft_1d(self, specs, n, win_len=None, hop=None, win=None):
        """Inverse 1D Short-Time Fourier Transform"""
        if win_len is None:
            win_len = self.win_len
        if hop is None:
            hop = self.hop
        if win is None:
            win = np.hanning(win_len)
            
        y = np.zeros(n)
        wsum = np.zeros(n)
        ptr = 0
        
        for start in range(0, n - win_len + 1, hop):
            frame = np.fft.irfft(specs[ptr]).real
            y[start:start+win_len] += frame * win
            wsum[start:start+win_len] += win**2
            ptr += 1
        
        wsum[wsum == 0] = 1.0
        return y / wsum
    
    def encode_spikes(self, z, thr=None, deriv_q=None):
        """SNN spike encoding"""
        if thr is None:
            thr = self.spike_thr
        if deriv_q is None:
            deriv_q = self.deriv_q
            
        s_thr = (z < thr).astype(np.int8)
        dz = np.diff(z, prepend=z[0])
        gamma = np.quantile(np.abs(dz), deriv_q)
        s_der = (dz < -gamma).astype(np.int8)
        
        return s_thr, s_der
    
    def lif_layer(self, stream, leak=None, thresholds=None, refrac=None):
        """Leaky Integrate-and-Fire layer"""
        if leak is None:
            leak = self.lif_leak
        if thresholds is None:
            thresholds = self.lif_thresholds
        if refrac is None:
            refrac = self.lif_refrac
            
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
    
    def fuzzy_membership(self, x, low, mid, high):
        """Triangular fuzzy membership function"""
        membership = np.zeros_like(x)
        
        # Left slope
        left_mask = (x >= low) & (x <= mid)
        membership[left_mask] = (x[left_mask] - low) / (mid - low)
        
        # Right slope
        right_mask = (x >= mid) & (x <= high)
        membership[right_mask] = (high - x[right_mask]) / (high - mid)
        
        return np.clip(membership, 0, 1)
    
    def extract_fuzzy_features(self, signal):
        """Extract fuzzy logic based features"""
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
        
        # Define fuzzy sets for signal amplitude
        low_amp = self.fuzzy_membership(signal_norm, -3, -1, 0)
        mid_amp = self.fuzzy_membership(signal_norm, -1, 0, 1)
        high_amp = self.fuzzy_membership(signal_norm, 0, 1, 3)
        
        # Define fuzzy sets for signal variability
        variability = np.abs(np.diff(signal_norm, prepend=signal_norm[0]))
        low_var = self.fuzzy_membership(variability, 0, 0.1, 0.5)
        mid_var = self.fuzzy_membership(variability, 0.1, 0.5, 1.0)
        high_var = self.fuzzy_membership(variability, 0.5, 1.0, 2.0)
        
        # Fuzzy features
        features = [
            np.mean(low_amp),   # Average low amplitude membership
            np.mean(mid_amp),   # Average mid amplitude membership
            np.mean(high_amp),  # Average high amplitude membership
            np.mean(low_var),   # Average low variability membership
            np.mean(mid_var),   # Average mid variability membership
            np.mean(high_var),  # Average high variability membership
            np.max(low_amp),    # Max low amplitude membership
            np.max(mid_amp),    # Max mid amplitude membership
            np.max(high_amp),   # Max high amplitude membership
            np.std(low_amp),    # Std of low amplitude membership
            np.std(mid_amp),    # Std of mid amplitude membership
            np.std(high_amp),   # Std of high amplitude membership
        ]
        
        return np.array(features)
    
    def process_lightcurve(self, time, flux):
        """Process a single light curve with audio-inspired techniques"""
        x = flux.copy()
        n = len(x)
        
        # Handle NaN values
        if np.any(np.isnan(x)):
            x = pd.Series(x).interpolate().fillna(method='bfill').fillna(method='ffill').values
        
        # 1) Pre-emphasis + Rolling Median
        x_pe = self.pre_emphasis(x)
        x_med = self.rolling_median(x_pe)
        
        # 2) STFT + Spectral Gating
        if len(x_med) >= self.win_len:
            S, centers, win = self.stft_1d(x_med)
            mag, phase = np.abs(S), np.angle(S)
            noise = np.percentile(mag, self.noise_pctl, axis=0, keepdims=True)
            mag_d = np.maximum(0.0, mag - self.gate_beta * noise)
            S_d = mag_d * np.exp(1j * phase)
            y_denoised = self.istft_1d(S_d, n)
        else:
            y_denoised = x_med.copy()
        
        # 3) De-emphasis
        a = self.pre_emph_alpha
        y_de = np.zeros_like(y_denoised)
        y_de[0] = y_denoised[0]
        for i in range(1, n):
            y_de[i] = y_denoised[i] + a * y_de[i-1]
        
        # 4) Normalize and blend
        y_norm = (y_de - np.median(y_de)) / (np.std(y_de) + 1e-9)
        x_norm = (x - np.median(x)) / (np.std(x) + 1e-9)
        y_final = 0.5 * y_norm + 0.5 * x_norm
        
        # 5) Novelty detection
        if len(x_med) >= self.win_len:
            spec_energy = mag_d.sum(axis=1)
            nov = np.maximum(0, np.diff(spec_energy, prepend=spec_energy[0]))
            nov_t = np.zeros_like(x, dtype=float)
            ptr = 0
            for start in range(0, n - self.win_len + 1, self.hop):
                center = start + self.win_len//2
                if center < len(nov_t):
                    nov_t[center] = nov[ptr] if ptr < len(nov) else 0
                ptr += 1
            nov_t = pd.Series(nov_t).rolling(window=min(self.win_len, len(nov_t)), 
                                           center=True, min_periods=1).mean().to_numpy()
        else:
            nov_t = np.zeros_like(x)
        
        # 6) SNN encoding + LIF
        z = (y_final - np.median(y_final)) / (np.std(y_final) + 1e-9)
        s_thr, s_der = self.encode_spikes(z)
        current = 0.8 * s_thr + 0.5 * s_der
        lif_spikes = self.lif_layer(current)
        
        # 7) Extract fuzzy features
        fuzzy_features = self.extract_fuzzy_features(y_final)
        
        return {
            'processed_flux': y_final,
            'novelty': nov_t,
            'spike_thr': s_thr,
            'spike_der': s_der,
            'lif_spikes': lif_spikes,
            'fuzzy_features': fuzzy_features,
            'original_flux': x,
            'denoised_flux': y_de
        }
    
    def extract_features_from_processed(self, processed_data):
        """Extract comprehensive features from processed light curve data"""
        features = []
        
        # Basic statistical features
        flux = processed_data['processed_flux']
        features.extend([
            np.mean(flux),
            np.std(flux),
            np.median(flux),
            np.percentile(flux, 25),
            np.percentile(flux, 75),
            np.min(flux),
            np.max(flux),
            np.ptp(flux),  # Peak-to-peak
            len(flux[flux > np.mean(flux) + 2*np.std(flux)]),  # Outliers
        ])
        
        # Novelty features
        novelty = processed_data['novelty']
        features.extend([
            np.mean(novelty),
            np.std(novelty),
            np.max(novelty),
            np.sum(novelty > np.percentile(novelty, 90)),  # High novelty events
        ])
        
        # SNN features
        s_thr = processed_data['spike_thr']
        s_der = processed_data['spike_der']
        lif_spikes = processed_data['lif_spikes']
        
        features.extend([
            np.sum(s_thr),
            np.sum(s_der),
            np.mean(s_thr),
            np.mean(s_der),
        ])
        
        # LIF features for each neuron
        for i in range(lif_spikes.shape[0]):
            features.extend([
                np.sum(lif_spikes[i]),
                np.mean(lif_spikes[i]),
            ])
        
        # Fuzzy features
        features.extend(processed_data['fuzzy_features'])
        
        return np.array(features)

class FuzzySVMExoplanetDetector:
    """
    Fuzzy Logic + Support Vector Machine based Exoplanet Detector
    """
    
    def __init__(self, C=1.0, gamma='scale', kernel='rbf'):
        self.preprocessor = AudioInspiredPreprocessor()
        self.svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, data):
        """Prepare features from dataset (koi_v1 or K2)"""
        print("Processing light curves with audio-inspired preprocessing...")
        
        # Check dataset format and process accordingly
        if 'kepid' in data.columns:
            # koi_v1 dataset format
            print("Processing koi_v1 dataset format...")
            all_features = []
            all_labels = []
            kepids = []
            
            for idx, row in data.iterrows():
                if idx % 500 == 0:
                    print(f"Processing sample {idx}/{len(data)}")
                
                # Extract flux columns (flux_0 to flux_499)
                flux_cols = [col for col in data.columns if col.startswith('flux_')]
                flux = row[flux_cols].values
                
                # Skip if flux is all zeros or NaN
                if np.all(flux == 0) or np.all(np.isnan(flux)):
                    continue
                
                # Create synthetic time array
                time = np.arange(len(flux))
                
                # Process the light curve
                processed = self.preprocessor.process_lightcurve(time, flux)
                
                # Extract features
                features = self.preprocessor.extract_features_from_processed(processed)
                
                all_features.append(features)
                all_labels.append(row['label'])
                kepids.append(row['kepid'])
            
            return np.array(all_features), np.array(all_labels), np.array(kepids)
            
        elif 'epic_id' in data.columns:
            # K2 dataset format (original code)
            print("Processing K2 dataset format...")
            grouped = data.groupby('epic_id')
            all_features = []
            all_labels = []
            epic_ids = []
            
            for epic_id, group in grouped:
                if len(group) < 50:  # Skip very short light curves
                    continue
                    
                time = group['time'].values
                flux = group['pdcsap_flux'].values
                label = group['label'].iloc[0]  # Assuming same label for all points in a light curve
                
                # Skip if flux is all zeros or NaN
                if np.all(flux == 0) or np.all(np.isnan(flux)):
                    continue
                
                # Process the light curve
                processed = self.preprocessor.process_lightcurve(time, flux)
                
                # Extract features
                features = self.preprocessor.extract_features_from_processed(processed)
                
                all_features.append(features)
                all_labels.append(label)
                epic_ids.append(epic_id)
            
            return np.array(all_features), np.array(all_labels), np.array(epic_ids)
        
        else:
            raise ValueError("Desteklenmeyen veri formatı. 'kepid' veya 'epic_id' sütunu bulunamadı.")
        
        # Generate feature names (same for both formats)
        self.feature_names = [
            'mean_flux', 'std_flux', 'median_flux', 'q25_flux', 'q75_flux',
            'min_flux', 'max_flux', 'ptp_flux', 'outliers_count',
            'novelty_mean', 'novelty_std', 'novelty_max', 'high_novelty_events',
            'spike_thr_sum', 'spike_der_sum', 'spike_thr_mean', 'spike_der_mean',
            'lif_neuron1_sum', 'lif_neuron1_mean',
            'lif_neuron2_sum', 'lif_neuron2_mean',
            'lif_neuron3_sum', 'lif_neuron3_mean',
            'fuzzy_low_amp_mean', 'fuzzy_mid_amp_mean', 'fuzzy_high_amp_mean',
            'fuzzy_low_var_mean', 'fuzzy_mid_var_mean', 'fuzzy_high_var_mean',
            'fuzzy_low_amp_max', 'fuzzy_mid_amp_max', 'fuzzy_high_amp_max',
            'fuzzy_low_amp_std', 'fuzzy_mid_amp_std', 'fuzzy_high_amp_std'
        ]
    
    def fit(self, X, y):
        """Fit the fuzzy SVM model"""
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training Fuzzy SVM model...")
        self.svm.fit(X_scaled, y)
        
        print("Model trained successfully!")
    
    def detect_exoplanets(self, X):
        """Detect exoplanets using fuzzy SVM"""
        X_scaled = self.scaler.transform(X)
        predictions = self.svm.predict(X_scaled)
        probabilities = self.svm.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities
    
    def evaluate_detection(self, X, y):
        """Evaluate exoplanet detection performance"""
        predictions, probabilities = self.detect_exoplanets(X)
        
        print("\n=== Fuzzy SVM Exoplanet Detection Evaluation ===")
        print(f"F1-Score: {f1_score(y, predictions):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y, probabilities):.4f}")
        
        # Calculate precision and recall
        precision, recall, f1_ensemble, _ = precision_recall_fscore_support(y, predictions, average='binary')
        
        print(f"\nDetection Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_ensemble:.4f}")
        
        return predictions, probabilities
    
    def plot_results(self, y_true, y_pred, probabilities):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Confusion Matrix (Fuzzy SVM)')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        axes[0,1].plot(fpr, tpr, label=f'ROC (AUC={roc_auc_score(y_true, probabilities):.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Feature importance (using SVM coefficients for linear kernel)
        if hasattr(self.svm, 'coef_') and self.svm.coef_ is not None:
            importances = np.abs(self.svm.coef_[0])
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            axes[1,0].barh(range(len(indices)), importances[indices])
            axes[1,0].set_yticks(range(len(indices)))
            axes[1,0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1,0].set_xlabel('Feature Importance (|SVM Coefficient|)')
            axes[1,0].set_title('Top 15 Feature Importances')
        else:
            axes[1,0].text(0.5, 0.5, 'Feature importance not available\nfor non-linear kernels', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Feature Importance')
        
        # Prediction Distribution
        axes[1,1].hist(probabilities[y_true == 0], bins=30, alpha=0.7, 
                      label='Non-Exoplanets', density=True)
        axes[1,1].hist(probabilities[y_true == 1], bins=30, alpha=0.7, 
                      label='Exoplanets', density=True)
        axes[1,1].set_xlabel('Prediction Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fuzzy_svm_audio_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Veri yükle - koi_v1.csv'yi öncelikle kontrol et
    print("Veri yükleniyor...")
    if Path("koi_v1.csv").exists():
        print("Loading koi_v1 dataset...")
        data = pd.read_csv('koi_v1.csv')
        dataset_name = "koi_v1"
        output_file = "koi_v1_audio_fuzzy_features.csv"
    elif Path("k2_dataset.csv").exists():
        print("Loading K2 dataset...")
        data = pd.read_csv('k2_dataset.csv')
        dataset_name = "K2"
        output_file = "k2_audio_fuzzy_features.csv"
    else:
        raise FileNotFoundError("Veri dosyası bulunamadı. koi_v1.csv veya k2_dataset.csv dosyalarından birini ekleyin.")
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Initialize detector
    detector = FuzzySVMExoplanetDetector()
    
    # Prepare features based on dataset type
    print(f"\nPreparing features with audio-inspired preprocessing for {dataset_name}...")
    X, y, ids = detector.prepare_features(data)
    
    print(f"Processed {len(X)} light curves")
    print(f"Feature shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    if len(X) == 0:
        print("No valid light curves found!")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    detector.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    predictions, probabilities = detector.evaluate_detection(X_test, y_test)
    
    # Plot results
    detector.plot_results(y_test, predictions, probabilities)
    
    # Save processed features
    feature_df = pd.DataFrame(X, columns=detector.feature_names)
    if dataset_name == "koi_v1":
        feature_df['kepid'] = ids
    else:
        feature_df['epic_id'] = ids
    feature_df['label'] = y
    feature_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to '{output_file}'")
    
    # Summary
    print(f"\n=== Final Results Summary ===")
    print(f"Dataset: {dataset_name} ({len(X)} light curves)")
    print(f"Exoplanets detected: {np.sum(predictions)}")
    print(f"Detection rate: {np.sum(predictions) / len(predictions):.4f}")
    print(f"True exoplanets: {np.sum(y_test)}")
    if np.sum(y_test) > 0:
        print(f"True positive rate: {np.sum((predictions == 1) & (y_test == 1)) / np.sum(y_test):.4f}")
    if np.sum(y_test == 0) > 0:
        print(f"False positive rate: {np.sum((predictions == 1) & (y_test == 0)) / np.sum(y_test == 0):.4f}")

if __name__ == "__main__":
    main()