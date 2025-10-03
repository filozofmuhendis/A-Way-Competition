import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def compare_models():
    # Veri hazırlama (önceki fonksiyonu kullan)
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Modeller
    models = {
        'CNN': 'Trained CNN model',  # Önceden eğitilmiş
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)
    }
    
    results = {}
    
    # Klasik ML modellerini eğit ve test et
    for name, model in models.items():
        if name != 'CNN':
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrikleri hesapla
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 0
            }
    
    # CNN sonuçlarını ekle (önceki eğitimden)
    results['CNN'] = final_metrics  # Önceki eğitimden
    
    # Sonuçları DataFrame'e çevir
    results_df = pd.DataFrame(results).T
    
    print("\n=== MODEL COMPARISON ===")
    print(results_df.round(4))
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    
    # Heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(results_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Model Performance Comparison')
    
    # Bar plot
    plt.subplot(2, 2, 2)
    results_df['f1_score'].plot(kind='bar')
    plt.title('F1-Score Comparison')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    
    # ROC-AUC comparison
    plt.subplot(2, 2, 3)
    results_df['roc_auc'].plot(kind='bar', color='green')
    plt.title('ROC-AUC Comparison')
    plt.ylabel('ROC-AUC')
    plt.xticks(rotation=45)
    
    # Precision vs Recall
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['precision'], results_df['recall'], s=100)
    for i, model in enumerate(results_df.index):
        plt.annotate(model, (results_df['precision'].iloc[i], results_df['recall'].iloc[i]))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df