import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# ML modeller
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

    # 2️⃣ Klasik ML modelleri eğit
    def train_ml_models(self):
        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=self.random_state),
            "DecisionTree": DecisionTreeClassifier(random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=150, random_state=self.random_state),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=self.random_state),
            "LogisticRegression": LogisticRegression(max_iter=500)
        }

        for name, model in classifiers.items():
            print(f"\n🚀 {name} eğitiliyor...")
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds)
            self.results[name] = {"accuracy": acc, "f1": f1}
            print(classification_report(self.y_test, preds))
            self.models[name] = model

    # 3️⃣ DL modelleri (örnek: ResNet-1D)
    def build_resnet(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        for _ in range(2):
            y = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
            y = layers.BatchNormalization()(y)
            x = layers.add([x, y])
            x = layers.ReLU()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_dl(self, model_name="ResNet"):
        X_train_dl = np.expand_dims(self.X_train, axis=-1)
        X_test_dl = np.expand_dims(self.X_test, axis=-1)
        y_train_dl = self.y_train.values if isinstance(self.y_train, pd.Series) else self.y_train
        y_test_dl = self.y_test.values if isinstance(self.y_test, pd.Series) else self.y_test

        model = self.build_resnet(input_shape=X_train_dl.shape[1:])
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_dl, y_train_dl,
            epochs=50, batch_size=64, validation_split=0.2,
            callbacks=[es], verbose=1
        )

        preds = (model.predict(X_test_dl) > 0.5).astype(int)
        acc = accuracy_score(y_test_dl, preds)
        f1 = f1_score(y_test_dl, preds)

        self.results[model_name] = {"accuracy": acc, "f1": f1}
        print(f"✅ {model_name} Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # 4️⃣ Sonuçları tabloya dök
    def summary(self):
        df_results = pd.DataFrame(self.results).T.sort_values(by="f1", ascending=False)
        print("\n📊 Model Karşılaştırma Sonuçları:\n")
        print(df_results)
        return df_results

if __name__ == "__main__":
    # CSV yükle
    df = pd.read_csv("k2_dataset_clean.csv")
    
    # Pipeline başlat
    pipe = ExoplanetPipeline(df=df, label_col="label")
    
    # Adımlar
    pipe.preprocess()
    pipe.train_ml_models()
    pipe.train_dl("ResNet")
    
    # Sonuç özeti
    results_df = pipe.summary()