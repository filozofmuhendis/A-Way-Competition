"""
Exoplanet Detection System Test Suite
Bu dosya bulanık mantık ve desteklemeli öğrenme sisteminin test edilmesi için oluşturulmuştur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from integrated_system import IntegratedExoplanetSystem
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Veri yükleme testini gerçekleştirir."""
    print("1. Veri Yükleme Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        print(f"   ✓ Veri başarıyla yüklendi: {len(system.data)} satır, {len(system.data.columns)} sütun")
        print(f"   ✓ Sütunlar: {list(system.data.columns)}")
        return True
    except Exception as e:
        print(f"   ✗ Veri yükleme hatası: {e}")
        return False

def test_fuzzy_system():
    """Bulanık mantık sistemi testini gerçekleştirir."""
    print("\n2. Bulanık Mantık Sistemi Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        
        # Bulanık mantık analizi
        fuzzy_results = system.run_fuzzy_analysis()
        print(f"   ✓ Bulanık mantık analizi tamamlandı: {len(fuzzy_results['predictions'])} sonuç")
        
        # Öznitelik istatistikleri - fuzzy_results artık bir dict, features DataFrame'i içeriyor
        features_df = fuzzy_results['features']  # normalized_features DataFrame'i
        feature_names = ['snr_transit', 'beta_factor', 'odd_even_diff', 'duty_cycle', 'depth_consistency']
        for feature in feature_names:
            if feature in features_df.columns:
                mean_val = features_df[feature].mean()
                std_val = features_df[feature].std()
                print(f"   ✓ {feature}: Ortalama={mean_val:.3f}, Std={std_val:.3f}")
        
        return True
    except Exception as e:
        print(f"   ✗ Bulanık mantık sistemi hatası: {e}")
        return False

def test_rl_environment():
    """Desteklemeli öğrenme ortamı testini gerçekleştirir."""
    print("\n3. Desteklemeli Öğrenme Ortamı Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        
        # Import AgentType enum
        from rl_environment import AgentType
        
        # Her ajan tipi için test
        agent_types = [AgentType.SIMPLE, AgentType.INTERMEDIATE, AgentType.COMPLEX]
        for agent_type in agent_types:
            env = system.multi_env.get_environment(agent_type)
            obs = env.reset()
            print(f"   ✓ {agent_type.value.capitalize()} ajan ortamı: Gözlem boyutu={len(obs)}")
            
            # Rastgele aksiyon testi
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"     - Aksiyon: {action}, Ödül: {reward:.3f}, Bitti: {done}")
        
        return True
    except Exception as e:
        print(f"   ✗ Desteklemeli öğrenme ortamı hatası: {e}")
        return False

def test_ppo_training():
    """PPO eğitimi testini gerçekleştirir."""
    print("\n4. PPO Eğitimi Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        
        # Kısa eğitim testi (sadece birkaç episode)
        print("   Kısa eğitim testi başlatılıyor...")
        system.train_rl_agents(episodes_per_agent=5)
        print("   ✓ PPO eğitimi başarıyla tamamlandı")
        
        return True
    except Exception as e:
        print(f"   ✗ PPO eğitimi hatası: {e}")
        return False

def test_system_evaluation():
    """Sistem değerlendirme testini gerçekleştirir."""
    print("\n5. Sistem Değerlendirme Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        
        # Sistem değerlendirmesi
        results = system.evaluate_system()
        print("   ✓ Sistem değerlendirmesi tamamlandı")
        
        # Sonuçları göster
        for agent_type, metrics in results.items():
            agent_name = str(agent_type).replace('AgentType.', '').upper()
            print(f"   {agent_name} Ajan:")
            for metric, value in metrics.items():
                if isinstance(value, tuple):
                    print(f"     - {metric}: {value[0]:.3f} ± {value[1]:.3f}")
                else:
                    print(f"     - {metric}: {value:.3f}")
        
        return True
    except Exception as e:
        print(f"   ✗ Sistem değerlendirme hatası: {e}")
        return False

def test_visualization():
    """Görselleştirme testini gerçekleştirir."""
    print("\n6. Görselleştirme Testi...")
    try:
        system = IntegratedExoplanetSystem('feature_extracted__k2_dataset_clean.csv')
        
        # Bulanık mantık görselleştirmesi
        system.fuzzy_system.visualize_membership_functions()
        print("   ✓ Bulanık mantık üyelik fonksiyonları görselleştirildi")
        
        # Sonuçları görselleştir
        system.visualize_results()
        print("   ✓ Sistem sonuçları görselleştirildi")
        
        return True
    except Exception as e:
        print(f"   ✗ Görselleştirme hatası: {e}")
        return False

def run_comprehensive_test():
    """Kapsamlı sistem testini çalıştırır."""
    print("=" * 60)
    print("EXOPLANET DETECTION SYSTEM - KAPSAMLI TEST")
    print("=" * 60)
    
    tests = [
        ("Veri Yükleme", test_data_loading),
        ("Bulanık Mantık Sistemi", test_fuzzy_system),
        ("Desteklemeli Öğrenme Ortamı", test_rl_environment),
        ("PPO Eğitimi", test_ppo_training),
        ("Sistem Değerlendirme", test_system_evaluation),
        ("Görselleştirme", test_visualization)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"   ✗ {test_name} testi beklenmedik hata: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SONUÇLARI: {passed_tests}/{total_tests} test başarılı")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 Tüm testler başarıyla geçildi!")
    else:
        print(f"⚠️  {total_tests - passed_tests} test başarısız oldu.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Ana test süitini çalıştır
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 Sistem hazır! Tam analiz için integrated_system.py dosyasını çalıştırabilirsiniz.")
    else:
        print("\n🔧 Bazı testler başarısız oldu. Lütfen hataları kontrol edin.")