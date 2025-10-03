# visualize.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_before_after(df_raw, df_norm, sample_idx=0, label_col="LABEL"):
    """
    Belirli bir örneğin (ör. sample_idx) flux değerlerini
    normalize edilmeden ve edildikten sonra yan yana çizer.
    """
    flux_cols = [c for c in df_raw.columns if c != label_col]

    raw_values = df_raw.iloc[sample_idx][flux_cols].values
    norm_values = df_norm.iloc[sample_idx][flux_cols].values

    plt.figure(figsize=(12, 5))

    # Orijinal
    plt.subplot(1, 2, 1)
    plt.plot(range(len(raw_values)), raw_values, 'b.-')
    plt.title(f"Before Normalization (Sample {sample_idx}, Label={df_raw.iloc[sample_idx][label_col]})")
    plt.xlabel("Flux Index")
    plt.ylabel("Flux Value")
    plt.grid(True)

    # Normalize edilmiş
    plt.subplot(1, 2, 2)
    plt.plot(range(len(norm_values)), norm_values, 'r.-')
    plt.title(f"After Normalization (Sample {sample_idx}, Label={df_norm.iloc[sample_idx][label_col]})")
    plt.xlabel("Flux Index")
    plt.ylabel("Normalized Flux")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
