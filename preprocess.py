# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, label_col="LABEL"):
    """
    Exoplanet flux verisini normalize eder.
    - Label kolonunu ayrı tutar
    - Flux kolonlarını z-score (ortalama=0, std=1) normalizasyonuna sokar
    """
    labels = df[label_col].values
    features = df.drop(columns=[label_col]).values
    
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    df_norm = pd.DataFrame(features_norm, columns=df.drop(columns=[label_col]).columns)
    df_norm[label_col] = labels
    return df_norm, scaler
