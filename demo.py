"""
Exoplanet Detection System Demo
Bu dosya sistemin temel işlevselliğini göstermek için oluşturulmuştur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from integrated_system import IntegratedExoplanetSystem
import warnings
warnings.filterwarnings('ignore')

def run_demo():
    """Sistem demo'sunu çalıştırır."""
    print("🌟 EXOPLANET DETECTION SYSTEM DEMO")
    print("=" * 50)
    
    try:
        # Sistem başlatma
        print("1. Sistem başlatılıyor...")
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        print("   ✓ Sistem başarıyla başlatıldı!")
        
        # Veri özeti - CSV dosyasını doğrudan oku
        data = pd.read_csv('feature_extracted__k2_dataset_clean.csv')
        print(f"\n2. Veri Özeti:")
        print(f"   - Toplam örnek: {len(data)}")
        print(f"   - Öznitelik sayısı: {len(data.columns)}")
        
        # Bulanık mantık analizi
        print("\n3. Bulanık Mantık Analizi...")
        fuzzy_results = system.run_fuzzy_analysis()
        print("   ✓ Bulanık mantık analizi tamamlandı!")
        
        # Öznitelik istatistikleri
        features = ['SNR_transit', 'beta_factor', 'odd_even_diff', 'duty_cycle', 'depth_consistency']
        print("\n   Öznitelik İstatistikleri:")
        if isinstance(fuzzy_results, dict):
            for feature in features:
                if feature in fuzzy_results:
                    values = fuzzy_results[feature]
                    if isinstance(values, (list, np.ndarray)):
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"   - {feature}: {mean_val:.3f} ± {std_val:.3f}")
        else:
            for feature in features:
                if hasattr(fuzzy_results, feature):
                    values = getattr(fuzzy_results, feature)
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"   - {feature}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Kısa eğitim
        print("\n4. Kısa PPO Eğitimi (Demo)...")
        system.train_agents(episodes=3, save_interval=10)
        print("   ✓ Demo eğitimi tamamlandı!")
        
        # Sistem değerlendirmesi
        print("\n5. Sistem Değerlendirmesi...")
        results = system.evaluate_system()
        print("   ✓ Değerlendirme tamamlandı!")
        
        # Görselleştirme
        print("\n6. Sonuçları Görselleştirme...")
        system.visualize_results()
        print("   ✓ Grafikler oluşturuldu!")
        
        print("\n🎉 Demo başarıyla tamamlandı!")
        print("\nDaha detaylı analiz için:")
        print("- python integrated_system.py (tam analiz)")
        print("- python test_system.py (sistem testleri)")
        
    except Exception as e:
        print(f"❌ Demo sırasında hata oluştu: {e}")
        print("Lütfen gerekli dosyaların mevcut olduğundan emin olun.")

if __name__ == "__main__":
    run_demo()