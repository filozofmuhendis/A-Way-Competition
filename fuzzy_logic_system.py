import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class ExoplanetFuzzySystem:
    """
    Exoplanet detection için bulanık mantık sistemi
    Beş öznitelik kullanarak exoplanet olasılığını hesaplar
    """
    
    def __init__(self):
        self.setup_fuzzy_variables()
        self.setup_membership_functions()
        self.setup_rules()
        self.setup_control_system()
        
    def setup_fuzzy_variables(self):
        """Bulanık değişkenleri tanımla"""
        # Giriş değişkenleri
        self.snr_transit = ctrl.Antecedent(np.arange(0, 101, 1), 'snr_transit')
        self.beta_factor = ctrl.Antecedent(np.arange(0, 101, 1), 'beta_factor')
        self.odd_even_diff = ctrl.Antecedent(np.arange(0, 101, 1), 'odd_even_diff')
        self.duty_cycle = ctrl.Antecedent(np.arange(0, 101, 1), 'duty_cycle')
        self.depth_consistency = ctrl.Antecedent(np.arange(0, 101, 1), 'depth_consistency')
        
        # Çıkış değişkeni
        self.exoplanet_probability = ctrl.Consequent(np.arange(0, 101, 1), 'exoplanet_probability')
        
    def setup_membership_functions(self):
        """Üyelik fonksiyonlarını tanımla"""
        
        # SNR Transit - Sinyal gücü
        self.snr_transit['düşük'] = fuzz.trimf(self.snr_transit.universe, [0, 0, 40])
        self.snr_transit['orta'] = fuzz.trimf(self.snr_transit.universe, [20, 50, 80])
        self.snr_transit['yüksek'] = fuzz.trimf(self.snr_transit.universe, [60, 100, 100])
        
        # Beta Faktörü - Veri kalitesi
        self.beta_factor['kötü'] = fuzz.trimf(self.beta_factor.universe, [0, 0, 40])
        self.beta_factor['orta'] = fuzz.trimf(self.beta_factor.universe, [20, 50, 80])
        self.beta_factor['iyi'] = fuzz.trimf(self.beta_factor.universe, [60, 100, 100])
        
        # Odd-Even Farkı - Düzenlilik
        self.odd_even_diff['düşük'] = fuzz.trimf(self.odd_even_diff.universe, [0, 0, 40])
        self.odd_even_diff['orta'] = fuzz.trimf(self.odd_even_diff.universe, [20, 50, 80])
        self.odd_even_diff['yüksek'] = fuzz.trimf(self.odd_even_diff.universe, [60, 100, 100])
        
        # Duty Cycle - Geometrik tutarlılık
        self.duty_cycle['düşük'] = fuzz.trimf(self.duty_cycle.universe, [0, 0, 40])
        self.duty_cycle['orta'] = fuzz.trimf(self.duty_cycle.universe, [20, 50, 80])
        self.duty_cycle['yüksek'] = fuzz.trimf(self.duty_cycle.universe, [60, 100, 100])
        
        # Depth Consistency - Sinyal kararlılığı
        self.depth_consistency['düşük'] = fuzz.trimf(self.depth_consistency.universe, [0, 0, 40])
        self.depth_consistency['orta'] = fuzz.trimf(self.depth_consistency.universe, [20, 50, 80])
        self.depth_consistency['yüksek'] = fuzz.trimf(self.depth_consistency.universe, [60, 100, 100])
        
        # Çıkış - Exoplanet olasılığı
        self.exoplanet_probability['çok_düşük'] = fuzz.trimf(self.exoplanet_probability.universe, [0, 0, 25])
        self.exoplanet_probability['düşük'] = fuzz.trimf(self.exoplanet_probability.universe, [10, 30, 50])
        self.exoplanet_probability['orta'] = fuzz.trimf(self.exoplanet_probability.universe, [30, 50, 70])
        self.exoplanet_probability['yüksek'] = fuzz.trimf(self.exoplanet_probability.universe, [50, 70, 90])
        self.exoplanet_probability['çok_yüksek'] = fuzz.trimf(self.exoplanet_probability.universe, [75, 100, 100])
        
    def create_rules(self):
        """Geliştirilmiş kapsamlı bulanık mantık kurallarını oluştur"""
        from enhanced_fuzzy_rules import EnhancedFuzzyRuleGenerator
        
        # Kural üreticiyi başlat
        rule_generator = EnhancedFuzzyRuleGenerator()
        
        # Dengeli kural seti üret (her kategoriden eşit sayıda)
        balanced_rules = rule_generator.generate_balanced_rules(rules_per_category=20)
        
        self.rules = []
        
        print(f"Geliştirilmiş kural tabanı oluşturuluyor: {len(balanced_rules)} kural")
        
        # Üretilen kuralları sisteme ekle
        for rule_info in balanced_rules:
            inputs = rule_info['inputs']
            output = rule_info['output']
            
            # Kural koşullarını oluştur
            conditions = []
            for feature, level in inputs.items():
                conditions.append(getattr(self, feature)[level])
            
            # Tüm koşulları AND ile birleştir
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition & condition
            
            # Kuralı ekle
            rule = ctrl.Rule(combined_condition, self.exoplanet_probability[output])
            self.rules.append(rule)
        
        # Kural dağılımını göster
        rule_distribution = {}
        for rule_info in balanced_rules:
            output = rule_info['output']
            rule_distribution[output] = rule_distribution.get(output, 0) + 1
        
        print("Kural dağılımı:")
        for output_level, count in rule_distribution.items():
            print(f"  {output_level}: {count} kural")
        
        print(f"Toplam kural sayısı: {len(self.rules)}")
    
    def setup_rules(self):
        """Bulanık mantık kurallarını tanımla - 5 öznitelik kullanarak"""
        self.create_rules()
        
    def setup_control_system(self):
        """Kontrol sistemini oluştur"""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def extract_features_from_data(self, df):
        """
        Ham veriden beş özniteliği çıkar
        """
        features = pd.DataFrame()
        
        # 1. SNR Transit - Sinyal gücü (bls_depth / flux_std oranı)
        features['snr_transit'] = df['bls_depth'] / (df['pdcsap_flux_std'] + 1e-8)
        
        # 2. Beta Faktörü - Veri kalitesi (flux_err / flux oranı)
        features['beta_factor'] = df['pdcsap_flux_err_std'] / (df['pdcsap_flux_std'] + 1e-8)
        
        # 3. Odd-Even Farkı - Düzenlilik (period'un yarısının modülü)
        features['odd_even_diff'] = np.abs(df['bls_period'] % 2.0)
        
        # 4. Duty Cycle - Geometrik tutarlılık (duration / period)
        features['duty_cycle'] = df['bls_duration'] / (df['bls_period'] + 1e-8)
        
        # 5. Depth Consistency - Sinyal kararlılığı (depth / flux_range)
        flux_range = df['pdcsap_flux_max'] - df['pdcsap_flux_min']
        features['depth_consistency'] = df['bls_depth'] / (flux_range + 1e-8)
        
        return features
    
    def normalize_features(self, features):
        """Özellikleri 0-100 aralığına normalize et"""
        scaler = MinMaxScaler(feature_range=(0, 100))
        normalized_features = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
        return normalized_features, scaler
    
    def predict(self, features):
        """Bulanık mantık sistemi ile tahmin yap"""
        predictions = []
        
        for idx, row in features.iterrows():
            try:
                # Giriş değerlerini ata
                self.simulation.input['snr_transit'] = row['snr_transit']
                self.simulation.input['beta_factor'] = row['beta_factor']
                self.simulation.input['odd_even_diff'] = row['odd_even_diff']
                self.simulation.input['duty_cycle'] = row['duty_cycle']
                self.simulation.input['depth_consistency'] = row['depth_consistency']
                
                # Hesapla
                self.simulation.compute()
                
                # Sonucu al - eğer output boşsa varsayılan değer kullan
                if 'exoplanet_probability' in self.simulation.output:
                    result = self.simulation.output['exoplanet_probability']
                else:
                    # Kurallar tetiklenmemişse orta değer kullan
                    result = 50.0
                    
                predictions.append(result)
                
            except Exception as e:
                print(f"Hata (satır {idx}): {e}")
                predictions.append(50.0)  # Varsayılan orta değer
                
        return np.array(predictions)
    
    def visualize_membership_functions(self):
        """Üyelik fonksiyonlarını görselleştir"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # SNR Transit
        self.snr_transit.view(ax=axes[0, 0])
        axes[0, 0].set_title('SNR Transit')
        
        # Beta Factor
        self.beta_factor.view(ax=axes[0, 1])
        axes[0, 1].set_title('Beta Faktörü')
        
        # Odd-Even Diff
        self.odd_even_diff.view(ax=axes[0, 2])
        axes[0, 2].set_title('Odd-Even Farkı')
        
        # Duty Cycle
        self.duty_cycle.view(ax=axes[1, 0])
        axes[1, 0].set_title('Duty Cycle')
        
        # Depth Consistency
        self.depth_consistency.view(ax=axes[1, 1])
        axes[1, 1].set_title('Depth Consistency')
        
        # Exoplanet Probability
        self.exoplanet_probability.view(ax=axes[1, 2])
        axes[1, 2].set_title('Exoplanet Olasılığı')
        
        plt.tight_layout()
        plt.savefig('fuzzy_membership_functions.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Ana fonksiyon - sistem testi"""
    # Veriyi yükle
    df = pd.read_csv('feature_extracted__k2_dataset_clean.csv')
    
    # Bulanık mantık sistemini oluştur
    fuzzy_system = ExoplanetFuzzySystem()
    
    # Özellikleri çıkar
    features = fuzzy_system.extract_features_from_data(df)
    print("Çıkarılan özellikler:")
    print(features.head())
    
    # Normalize et
    normalized_features, scaler = fuzzy_system.normalize_features(features)
    print("\nNormalize edilmiş özellikler:")
    print(normalized_features.head())
    
    # Tahmin yap
    predictions = fuzzy_system.predict(normalized_features)
    
    # Sonuçları göster
    results_df = pd.DataFrame({
        'id': df['id'],
        'gerçek_label': df['label'],
        'bulanık_tahmin': predictions / 100.0,  # 0-1 aralığına çevir
        'snr_transit': normalized_features['snr_transit'],
        'beta_factor': normalized_features['beta_factor'],
        'odd_even_diff': normalized_features['odd_even_diff'],
        'duty_cycle': normalized_features['duty_cycle'],
        'depth_consistency': normalized_features['depth_consistency']
    })
    
    print("\nİlk 10 tahmin:")
    print(results_df.head(10))
    
    # Performans değerlendirmesi
    from sklearn.metrics import accuracy_score, classification_report
    
    # Bulanık çıktıyı binary'ye çevir (eşik: 0.5)
    binary_predictions = (predictions / 100.0 > 0.5).astype(int)
    
    accuracy = accuracy_score(df['label'], binary_predictions)
    print(f"\nBulanık mantık sistemi doğruluğu: {accuracy:.4f}")
    
    print("\nDetaylı performans raporu:")
    print(classification_report(df['label'], binary_predictions))
    
    # Üyelik fonksiyonlarını görselleştir
    fuzzy_system.visualize_membership_functions()
    
    return fuzzy_system, results_df

if __name__ == "__main__":
    fuzzy_system, results = main()