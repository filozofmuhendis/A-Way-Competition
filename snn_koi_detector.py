import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class SNNKoiDetector:
    """
    Spiking Neural Network inspired Exoplanet Detector for KOI dataset
    Adapted for pre-processed KOI data with statistical features and flux values
    """
    
    def __init__(self, threshold=0.5, leak_factor=0.9, spike_threshold=0.3):
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.spike_threshold = spike_threshold
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_names = []
        
    def detrend_signal(self, signal):
        """Remove linear trend from signal"""
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend = np.polyval(coeffs, x)
        return signal - trend
    
    def spike_encoding(self, signal, threshold=None):
        """Convert analog signal to spike train"""
        if threshold is None:
            threshold = self.spike_threshold
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Generate spikes where signal exceeds threshold
        spikes = np.zeros_like(signal_norm)
        spikes[np.abs(signal_norm) > threshold] = 1
        
        return spikes
    
    def lif_layer(self, spikes, tau=10.0, v_reset=0.0, v_threshold=1.0):
        """Leaky Integrate-and-Fire neuron layer"""
        membrane_potential = np.zeros_like(spikes, dtype=float)
        output_spikes = np.zeros_like(spikes)
        
        v = v_reset
        for i in range(len(spikes)):
            # Leak
            v *= self.leak_factor
            # Integrate
            v += spikes[i]
            
            # Fire
            if v >= v_threshold:
                output_spikes[i] = 1
                v = v_reset
            
            membrane_potential[i] = v
        
        return output_spikes, membrane_potential
    
    def extract_snn_features(self, flux_data):
        """Extract SNN-inspired features from flux data"""
        features = []
        
        # Detrend the flux data
        detrended = self.detrend_signal(flux_data)
        
        # Spike encoding
        spikes = self.spike_encoding(detrended)
        
        # LIF processing
        lif_spikes, membrane_potential = self.lif_layer(spikes)
        
        # Feature extraction
        features.extend([
            np.sum(spikes),  # Total spike count
            np.mean(spikes),  # Average spike rate
            np.std(spikes),   # Spike rate variability
            np.sum(lif_spikes),  # LIF output spike count
            np.mean(membrane_potential),  # Average membrane potential
            np.std(membrane_potential),   # Membrane potential variability
            np.max(membrane_potential),   # Peak membrane potential
            np.min(membrane_potential),   # Minimum membrane potential
        ])
        
        # Windowed analysis
        window_size = len(flux_data) // 10
        if window_size > 0:
            for i in range(0, len(flux_data), window_size):
                window = flux_data[i:i+window_size]
                if len(window) > 5:
                    window_spikes = self.spike_encoding(window)
                    features.extend([
                        np.sum(window_spikes),
                        np.std(window_spikes) if len(window_spikes) > 1 else 0
                    ])
        
        return np.array(features)
    
    def extract_statistical_features(self, row):
        """Extract additional statistical features from the data row"""
        # Get flux columns (flux_0 to flux_499)
        flux_cols = [col for col in row.index if col.startswith('flux_')]
        flux_data = row[flux_cols].values
        
        # Basic statistical features (already in dataset but we can compute more)
        features = []
        
        # Existing statistical features
        features.extend([
            row['mean_flux'],
            row['min_flux'],
            row['max_flux'],
            row['std_flux'],
            row['mad_flux'],
            row['SAP'],
            row['PDCSAP'],
            row['SAP_PDCSAP']
        ])
        
        # Additional statistical features
        features.extend([
            np.median(flux_data),
            np.percentile(flux_data, 25),
            np.percentile(flux_data, 75),
            np.percentile(flux_data, 90),
            np.percentile(flux_data, 10),
            np.var(flux_data),
            np.ptp(flux_data),  # Peak-to-peak
            len(flux_data[flux_data > np.mean(flux_data) + 2*np.std(flux_data)]),  # Outliers
            len(flux_data[flux_data < np.mean(flux_data) - 2*np.std(flux_data)]),  # Negative outliers
        ])
        
        return np.array(features)
    
    def extract_features(self, data):
        """Extract comprehensive features from KOI dataset"""
        all_features = []
        
        print("Extracting features from KOI dataset...")
        for idx, row in data.iterrows():
            if idx % 500 == 0:
                print(f"Processing row {idx}/{len(data)}")
            
            # Get flux data
            flux_cols = [col for col in row.index if col.startswith('flux_')]
            flux_data = row[flux_cols].values
            
            # Extract statistical features
            stat_features = self.extract_statistical_features(row)
            
            # Extract SNN features
            snn_features = self.extract_snn_features(flux_data)
            
            # Combine all features
            combined_features = np.concatenate([stat_features, snn_features])
            all_features.append(combined_features)
        
        features_array = np.array(all_features)
        
        # Generate feature names
        self.feature_names = [
            'mean_flux', 'min_flux', 'max_flux', 'std_flux', 'mad_flux',
            'SAP', 'PDCSAP', 'SAP_PDCSAP', 'median_flux', 'q25_flux',
            'q75_flux', 'q90_flux', 'q10_flux', 'var_flux', 'ptp_flux',
            'outliers_pos', 'outliers_neg', 'total_spikes', 'avg_spike_rate',
            'spike_variability', 'lif_spikes', 'avg_membrane_potential',
            'membrane_variability', 'peak_membrane', 'min_membrane'
        ]
        
        # Add windowed features
        window_features = []
        for i in range(10):  # 10 windows
            window_features.extend([f'window_{i}_spikes', f'window_{i}_spike_std'])
        self.feature_names.extend(window_features)
        
        return features_array
    
    def fit(self, X, y):
        """Fit multiple models on the extracted features"""
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training models...")
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['logistic'].fit(X_scaled, y)
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.models['random_forest'].fit(X_scaled, y)
        
        # SVM
        self.models['svm'] = SVC(probability=True, random_state=42, class_weight='balanced')
        self.models['svm'].fit(X_scaled, y)
        
        # Isolation Forest for anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.models['isolation_forest'].fit(X_scaled[y == 0])  # Fit on non-exoplanets
        
        print("Models trained successfully!")
    
    def detect_exoplanets(self, X):
        """Detect exoplanets using ensemble of models"""
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                # Isolation forest returns -1 for anomalies, 1 for normal
                pred = model.predict(X_scaled)
                predictions[name] = (pred == -1).astype(int)
                # Convert to probabilities (rough approximation)
                scores = model.decision_function(X_scaled)
                probabilities[name] = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            else:
                predictions[name] = model.predict(X_scaled)
                probabilities[name] = model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble prediction (majority vote)
        ensemble_pred = np.zeros(len(X))
        for pred in predictions.values():
            ensemble_pred += pred
        ensemble_pred = (ensemble_pred >= len(self.models) / 2).astype(int)
        
        return ensemble_pred, predictions, probabilities
    
    def evaluate_detection(self, X, y):
        """Evaluate exoplanet detection performance"""
        predictions, individual_preds, probabilities = self.detect_exoplanets(X)
        
        print("\n=== Exoplanet Detection Evaluation ===")
        print(f"Ensemble F1-Score: {f1_score(y, predictions):.4f}")
        print(f"Ensemble ROC-AUC: {roc_auc_score(y, np.mean(list(probabilities.values()), axis=0)):.4f}")
        
        print("\nIndividual Model Performance:")
        for name in self.models.keys():
            if name in individual_preds:
                f1 = f1_score(y, individual_preds[name])
                if name != 'isolation_forest':
                    auc = roc_auc_score(y, probabilities[name])
                    print(f"{name}: F1={f1:.4f}, ROC-AUC={auc:.4f}")
                else:
                    print(f"{name}: F1={f1:.4f}")
        
        # Calculate precision and recall for ensemble
        precision, recall, f1_ensemble, _ = precision_recall_fscore_support(y, predictions, average='binary')
        
        print(f"\nEnsemble Detection Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_ensemble:.4f}")
        
        return predictions, probabilities
    
    def plot_results(self, y_true, y_pred, probabilities):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
        axes[0,0].set_title('Confusion Matrix (Ensemble)')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC Curves
        from sklearn.metrics import roc_curve
        for name, probs in probabilities.items():
            if name != 'isolation_forest':
                fpr, tpr, _ = roc_curve(y_true, probs)
                axes[0,1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_true, probs):.3f})')
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Feature Importance (Random Forest)
        if 'random_forest' in self.models:
            importances = self.models['random_forest'].feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            axes[1,0].barh(range(len(indices)), importances[indices])
            axes[1,0].set_yticks(range(len(indices)))
            axes[1,0].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1,0].set_xlabel('Feature Importance')
            axes[1,0].set_title('Top 15 Feature Importances (Random Forest)')
        
        # Prediction Distribution
        axes[1,1].hist(probabilities['logistic'][y_true == 0], bins=30, alpha=0.7, 
                      label='Non-Exoplanets', density=True)
        axes[1,1].hist(probabilities['logistic'][y_true == 1], bins=30, alpha=0.7, 
                      label='Exoplanets', density=True)
        axes[1,1].set_xlabel('Prediction Probability')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('koi_snn_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Loading KOI dataset...")
    data = pd.read_csv('koi_v1.csv')
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Prepare features and labels
    X_data = data.drop(['kepid', 'label'], axis=1)
    y = data['label'].values
    
    print(f"Features shape: {X_data.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Exoplanet ratio: {np.mean(y):.4f}")
    
    # Initialize detector
    detector = SNNKoiDetector()
    
    # Extract features
    print("\nExtracting SNN and statistical features...")
    X_features = detector.extract_features(X_data)
    
    print(f"Extracted features shape: {X_features.shape}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
        print(f"Fold {fold + 1}/5")
        
        X_train, X_val = X_features[train_idx], X_features[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create temporary detector for this fold
        fold_detector = SNNKoiDetector()
        fold_detector.feature_names = detector.feature_names
        fold_detector.fit(X_train, y_train)
        
        # Evaluate
        predictions, _, probabilities = fold_detector.detect_exoplanets(X_val)
        f1 = f1_score(y_val, predictions)
        cv_scores.append(f1)
        print(f"Fold {fold + 1} Detection F1-Score: {f1:.4f}")
    
    print(f"\nCross-validation detection results:")
    print(f"Average Detection F1-Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    detector.fit(X_features, y)
    
    # Final evaluation
    print("\nFinal exoplanet detection evaluation:")
    predictions, probabilities = detector.evaluate_detection(X_features, y)
    
    # Plot results
    detector.plot_results(y, predictions, probabilities)
    
    # Save features
    feature_df = pd.DataFrame(X_features, columns=detector.feature_names)
    feature_df['kepid'] = data['kepid']
    feature_df['label'] = y
    feature_df.to_csv('koi_snn_features.csv', index=False)
    print("\nFeatures saved to 'koi_snn_features.csv'")
    
    # Summary statistics
    print(f"\n=== Final Results Summary ===")
    print(f"Dataset: KOI v1 ({len(data)} samples)")
    print(f"Exoplanets detected: {np.sum(predictions)}")
    print(f"Detection rate: {np.sum(predictions) / len(predictions):.4f}")
    print(f"True exoplanets: {np.sum(y)}")
    print(f"True positive rate: {np.sum((predictions == 1) & (y == 1)) / np.sum(y):.4f}")
    print(f"False positive rate: {np.sum((predictions == 1) & (y == 0)) / np.sum(y == 0):.4f}")

if __name__ == "__main__":
    main()