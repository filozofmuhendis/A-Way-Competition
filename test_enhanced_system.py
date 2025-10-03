"""
Geliştirilmiş Fuzzy Logic + PPO Sistemi Test Scripti
Yeni kural tabanı ve özelleştirilmiş PPO ajanlarını test eder
"""

import sys
import traceback
import numpy as np
from integrated_system import IntegratedExoplanetSystem
from enhanced_fuzzy_rules import EnhancedFuzzyRuleGenerator
from enhanced_ppo_agents import FuzzyOutputType, SpecializedPPOAgent

def test_enhanced_fuzzy_rules():
    """Geliştirilmiş fuzzy kurallarını test et"""
    print("=" * 60)
    print("GELİŞTİRİLMİŞ FUZZY KURALLAR TESTİ")
    print("=" * 60)
    
    try:
        # Rule generator oluştur
        generator = EnhancedFuzzyRuleGenerator()
        
        # Tüm kuralları oluştur
        all_rules = generator.generate_all_rules()
        print(f"✓ Toplam {len(all_rules)} kural oluşturuldu")
        
        # Kural dağılımını kontrol et
        rule_distribution = {}
        for rule in all_rules:
            output_level = rule['output']  # 'output_level' yerine 'output' kullan
            rule_distribution[output_level] = rule_distribution.get(output_level, 0) + 1
        
        print("\nKural Dağılımı:")
        for level, count in rule_distribution.items():
            print(f"  {level}: {count} kural")
        
        # Optimize edilmiş kuralları test et
        optimized_rules = generator.generate_optimized_rules(50)
        print(f"\n✓ {len(optimized_rules)} optimize edilmiş kural oluşturuldu")
        
        # Balanced kuralları test et
        balanced_rules = generator.generate_balanced_rules(20)
        print(f"✓ {len(balanced_rules)} dengeli kural oluşturuldu")
        
        return True
        
    except Exception as e:
        print(f"✗ Geliştirilmiş fuzzy kurallar testi başarısız: {e}")
        traceback.print_exc()
        return False

def test_specialized_ppo_agents():
    """Özelleştirilmiş PPO ajanlarını test et"""
    print("\n" + "=" * 60)
    print("ÖZELLEŞTİRİLMİŞ PPO AJANLARI TESTİ")
    print("=" * 60)
    
    try:
        # Her fuzzy output type için ajan oluştur
        agents = {}
        
        for fuzzy_output in FuzzyOutputType:
            print(f"\n{fuzzy_output.value} için ajan oluşturuluyor...")
            
            agent = SpecializedPPOAgent(
                state_dim=6,
                action_dim=2,
                fuzzy_output_type=fuzzy_output
            )
            
            agents[fuzzy_output.value] = agent
            print(f"✓ {fuzzy_output.value} ajanı oluşturuldu")
            
            # Ağ mimarisini test et
            test_state = np.random.randn(6)
            action = agent.select_action(test_state)
            print(f"  Test aksiyonu: {action}")
        
        print(f"\n✓ Toplam {len(agents)} özelleştirilmiş ajan oluşturuldu")
        return True
        
    except Exception as e:
        print(f"✗ Özelleştirilmiş PPO ajanları testi başarısız: {e}")
        traceback.print_exc()
        return False

def test_enhanced_integrated_system():
    """Geliştirilmiş entegre sistemi test et"""
    print("\n" + "=" * 60)
    print("GELİŞTİRİLMİŞ ENTEGRE SİSTEM TESTİ")
    print("=" * 60)
    
    try:
        # Geliştirilmiş sistem oluştur
        print("Geliştirilmiş entegre sistem oluşturuluyor...")
        system = IntegratedExoplanetSystem(use_enhanced_system=True)
        
        print("✓ Geliştirilmiş sistem başarıyla oluşturuldu")
        
        # Sistem istatistiklerini al
        stats = system.get_enhanced_system_stats()
        if stats:
            print(f"\nSistem İstatistikleri:")
            print(f"  Toplam özelleştirilmiş ajan: {stats['total_specialized_agents']}")
            print(f"  Fuzzy çıktı tipi sayısı: {stats['fuzzy_output_types']}")
            print(f"  Ajan tipi sayısı: {stats['agent_types']}")
            
            print(f"\nFuzzy çıktı tipi başına ajan sayısı:")
            for fuzzy_output, count in stats['agents_per_fuzzy_output'].items():
                print(f"  {fuzzy_output}: {count} ajan")
        
        # Fuzzy analizi test et
        print("\nFuzzy analizi test ediliyor...")
        fuzzy_results = system.run_fuzzy_analysis()
        print(f"✓ Fuzzy analizi tamamlandı - Doğruluk: {fuzzy_results['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geliştirilmiş entegre sistem testi başarısız: {e}")
        traceback.print_exc()
        return False

def test_enhanced_training():
    """Geliştirilmiş eğitim sürecini test et"""
    print("\n" + "=" * 60)
    print("GELİŞTİRİLMİŞ EĞİTİM SÜRECİ TESTİ")
    print("=" * 60)
    
    try:
        # Sistem oluştur
        system = IntegratedExoplanetSystem(use_enhanced_system=True)
        
        # Kısa eğitim testi (5 episode)
        print("Kısa eğitim testi başlatılıyor (5 episode)...")
        enhanced_results = system.train_rl_agents(episodes_per_agent=5, use_enhanced=True)
        
        print("✓ Geliştirilmiş eğitim tamamlandı")
        
        # Sonuçları kontrol et
        if enhanced_results:
            print(f"\nEğitim Sonuçları:")
            for agent_key, results in enhanced_results.items():
                print(f"  {agent_key}: {results['final_avg_reward']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geliştirilmiş eğitim testi başarısız: {e}")
        traceback.print_exc()
        return False

def test_system_comparison():
    """Sistem karşılaştırması testi"""
    print("\n" + "=" * 60)
    print("SİSTEM KARŞILAŞTIRMASI TESTİ")
    print("=" * 60)
    
    try:
        # Sistem oluştur
        system = IntegratedExoplanetSystem(use_enhanced_system=True)
        
        # Fuzzy analizi
        fuzzy_results = system.run_fuzzy_analysis()
        
        # Standart sistem eğitimi
        print("\nStandart sistem eğitimi (3 episode)...")
        standard_results = system.train_rl_agents(episodes_per_agent=3, use_enhanced=False)
        
        # Geliştirilmiş sistem eğitimi
        print("\nGeliştirilmiş sistem eğitimi (3 episode)...")
        enhanced_results = system.train_rl_agents(episodes_per_agent=3, use_enhanced=True)
        
        # Karşılaştırma
        system.compare_training_results(standard_results, enhanced_results)
        
        print("✓ Sistem karşılaştırması tamamlandı")
        return True
        
    except Exception as e:
        print(f"✗ Sistem karşılaştırması testi başarısız: {e}")
        traceback.print_exc()
        return False

def run_all_enhanced_tests():
    """Tüm geliştirilmiş sistem testlerini çalıştır"""
    print("GELİŞTİRİLMİŞ EXOPLANET DETECTION SİSTEMİ - KAPSAMLI TEST")
    print("=" * 80)
    
    tests = [
        ("Geliştirilmiş Fuzzy Kurallar", test_enhanced_fuzzy_rules),
        ("Özelleştirilmiş PPO Ajanları", test_specialized_ppo_agents),
        ("Geliştirilmiş Entegre Sistem", test_enhanced_integrated_system),
        ("Geliştirilmiş Eğitim Süreci", test_enhanced_training),
        ("Sistem Karşılaştırması", test_system_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ BAŞARILI" if result else "✗ BAŞARISIZ"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n{test_name}: ✗ BAŞARISIZ - {e}")
    
    # Özet rapor
    print("\n" + "=" * 80)
    print("TEST SONUÇLARI ÖZETİ")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ BAŞARILI" if result else "✗ BAŞARISIZ"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nToplam: {passed}/{total} test başarılı")
    
    if passed == total:
        print("\n🎉 TÜM TESTLER BAŞARILI! Geliştirilmiş sistem hazır.")
    else:
        print(f"\n⚠️  {total-passed} test başarısız. Lütfen hataları kontrol edin.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_enhanced_tests()
    sys.exit(0 if success else 1)