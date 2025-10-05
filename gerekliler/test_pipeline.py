import pandas as pd
import sys
import os
sys.path.append('..')

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# DL
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

class ExoplanetPipeline:
    def __init__(self, df, label_col, test_size=0.2, random_state=42):
        self.df = df
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    # 1️⃣ Veri hazırlama
    def preprocess(self):
        X = self.df.drop(columns=[self.label_col])
        y = self.df[self.label_col]

        # SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"✅ SMOTE sonrası veri boyutu: {X_res.shape}")

        # Train/Test
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=self.test_size, random_state=self.random_state
        )

        # Normalize
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

# Test the class
if __name__ == "__main__":
    # CSV yükle
    df = pd.read_csv("k2_dataset_clean.csv")
    
    # Pipeline başlat
    pipe = ExoplanetPipeline(df=df, label_col="label")
    print("✅ ExoplanetPipeline başarıyla oluşturuldu!")
    
    # Test preprocess
    pipe.preprocess()
    print("✅ Preprocess başarıyla tamamlandı!")