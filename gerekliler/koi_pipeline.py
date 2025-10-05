import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import os

# İsteğe bağlı kütüphaneler - yoksa atla
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Uyarı: LightGBM kütüphanesi bulunamadı, LightGBM modeli atlanacak.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Uyarı: XGBoost kütüphanesi bulunamadı, XGBoost modeli atlanacak.")


def sigma_clip(df, sigma=3):
    """
    3 sigma kuralı: her sütun için ortalama ± 3*std aralığı dışında kalan değerler kırpılır
    """
    df_clipped = df.copy()
    for col in df_clipped.select_dtypes(include=[np.number]).columns:
        mean = df_clipped[col].mean()
        std = df_clipped[col].std()
        upper = mean + sigma * std
        lower = mean - sigma * std
        df_clipped[col] = np.clip(df_clipped[col], lower, upper)
    return df_clipped


def preprocess_data(csv_path, label_col="label"):
    """
    Veri ön işleme: sigma clipping, normalizasyon ve SMOTE uygular
    """
    print("=== 1. Veri setini oku ===")
    df = pd.read_csv(csv_path)
    
    print("=== 2. Hedef (etiket) sütununu belirle ===")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("=== 3. Sigma Clipping uygula ===")
    X_clipped = sigma_clip(X, sigma=3)
    
    print("=== 4. Normalizasyon (StandardScaler) ===")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clipped), columns=X_clipped.columns)
    
    print("=== 5. SMOTE (class imbalance düzeltme) ===")
    print("SMOTE öncesi dağılım:", Counter(y))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    print("SMOTE sonrası dağılım:", Counter(y_res))
    
    print("=== 6. Yeni veri setini kaydet ===")
    output_path = csv_path.replace(".csv", "_sigma_norm_smote.csv")
    df_processed = pd.concat([X_res, y_res], axis=1)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Yeni veri seti başarıyla kaydedildi: {output_path}")
    return output_path


def train_and_evaluate_models(processed_csv_path, label_col="label"):
    """
    Makine öğrenmesi modellerini eğitir ve değerlendirir
    """
    print("\n=== MODEL EĞİTİMİ VE DEĞERLENDİRME ===")
    
    print("=== 1) Veri seti ===")
    df = pd.read_csv(processed_csv_path)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Kategorik etiketleri sayısallaştır
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
    
    print("=== 2) Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("=== 3) Modeller ===")
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    
    # İsteğe bağlı modelleri ekle
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(random_state=42)
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    
    print("=== 4) Eğitim + Değerlendirme ===")
    rows = []
    reports_dir = os.path.join(os.path.dirname(processed_csv_path), "model_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"Eğitiliyor: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Makro & Ağırlıklı metrikler
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        
        acc = accuracy_score(y_test, y_pred)
        
        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision_macro": prec_macro,
            "Recall_macro": rec_macro,
            "F1_macro": f1_macro,
            "Precision_weighted": prec_weight,
            "Recall_weighted": rec_weight,
            "F1_weighted": f1_weight,
        })
        
        # Sınıf-bazlı rapor (.txt)
        report_txt = classification_report(y_test, y_pred, zero_division=0)
        with open(os.path.join(reports_dir, f"{name.replace(' ', '_')}_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_txt)
        
        # Confusion matrix (.csv)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(os.path.join(reports_dir, f"{name.replace(' ', '_')}_confusion_matrix.csv"), index=False)
    
    print("=== 5) Sonuç tablosu ===")
    results_df = pd.DataFrame(rows).sort_values(by="F1_macro", ascending=False)
    out_csv = os.path.join(os.path.dirname(processed_csv_path), "v1_model_results_with_recall_f1.csv")
    results_df.to_csv(out_csv, index=False)
    
    print(results_df)
    print(f"\nDetaylı raporlar: {reports_dir}")
    print(f"Özet tablo: {out_csv}")
    
    return results_df


def main():
    """
    Ana fonksiyon - tüm pipeline'ı çalıştırır
    """
    # Veri dosyası yolu - mevcut CSV dosyalarını kontrol et
    possible_files = ["v1.csv", "koi_v1.csv", "k2_dataset_clean.csv"]
    input_csv = None
    
    for file in possible_files:
        if os.path.exists(file):
            input_csv = file
            break
    
    if input_csv is None:
        print("Hata: Uygun veri dosyası bulunamadı!")
        print(f"Aranan dosyalar: {possible_files}")
        print("Lütfen bu dosyalardan birinin mevcut dizinde olduğundan emin olun.")
        return
    
    print(f"Kullanılacak veri dosyası: {input_csv}")
    label_column = "label"
    
    try:
        # 1. Veri ön işleme
        processed_csv = preprocess_data(input_csv, label_column)
        
        # 2. Model eğitimi ve değerlendirme
        results = train_and_evaluate_models(processed_csv, label_column)
        
        print("\n=== PİPELİNE TAMAMLANDI ===")
        print("En iyi performans gösteren modeller:")
        print(results.head(3)[["Model", "Accuracy", "F1_macro"]])
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        print("Lütfen veri dosyasının formatını ve sütun isimlerini kontrol edin.")


if __name__ == "__main__":
    main()