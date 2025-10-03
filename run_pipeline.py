"""
Exoplanet Detection Pipeline Runner
===================================

Bu script, Exoplanet Detection Pipeline'ını farklı CSV dosyaları 
ve konfigürasyonlarla çalıştırmak için kullanılır.

Kullanım:
    python run_pipeline.py --csv data.csv --output results/
    python run_pipeline.py --csv data.csv --config custom_config.json
    python run_pipeline.py --help
"""

import argparse
import json
import os
import sys
from datetime import datetime

from exoplanet_detection_pipeline import ExoplanetDetectionPipeline


def load_config(config_path: str) -> dict:
    """Konfigürasyon dosyasını yükle"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Konfigürasyon yükleme hatası: {e}")
        return None


def create_sample_config(output_path: str):
    """Örnek konfigürasyon dosyası oluştur"""
    sample_config = {
        "fuzzy_system": {
            "enable_enhanced_rules": True,
            "rule_count_target": 200
        },
        "ppo_system": {
            "enable_specialized_agents": True,
            "episodes_per_combination": 50,
            "enable_training": True
        },
        "preprocessing": {
            "enable_audio_features": True,
            "normalize_features": True,
            "handle_missing_values": True
        },
        "output": {
            "save_intermediate_results": True,
            "create_visualizations": True,
            "detailed_reports": True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Örnek konfigürasyon oluşturuldu: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exoplanet Detection Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run_pipeline.py --csv feature_extracted__k2_dataset_clean.csv
  python run_pipeline.py --csv data.csv --output my_results/
  python run_pipeline.py --csv data.csv --config custom_config.json
  python run_pipeline.py --create-config sample_config.json
        """
    )
    
    parser.add_argument(
        '--csv', 
        type=str, 
        help='İşlenecek CSV dosyasının yolu'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Çıktı dizini (varsayılan: otomatik oluşturulur)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Konfigürasyon dosyası yolu (JSON formatında)'
    )
    
    parser.add_argument(
        '--create-config', 
        type=str, 
        help='Örnek konfigürasyon dosyası oluştur'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Hızlı mod (PPO eğitimini atla)'
    )
    
    parser.add_argument(
        '--no-viz', 
        action='store_true',
        help='Görselleştirmeleri atla'
    )
    
    args = parser.parse_args()
    
    # Örnek konfigürasyon oluştur
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # CSV dosyası kontrolü
    if not args.csv:
        print("❌ CSV dosyası belirtilmedi. --csv parametresini kullanın.")
        parser.print_help()
        return
    
    if not os.path.exists(args.csv):
        print(f"❌ CSV dosyası bulunamadı: {args.csv}")
        return
    
    # Konfigürasyonu yükle
    config = None
    if args.config:
        config = load_config(args.config)
        if config is None:
            return
    else:
        # Varsayılan konfigürasyon
        config = {}
    
    # Hızlı mod ayarları
    if args.quick:
        if 'ppo_system' not in config:
            config['ppo_system'] = {}
        config['ppo_system']['enable_training'] = False
        print("🚀 Hızlı mod aktif - PPO eğitimi atlanacak")
    
    # Görselleştirme ayarları
    if args.no_viz:
        if 'output' not in config:
            config['output'] = {}
        config['output']['create_visualizations'] = False
        print("📊 Görselleştirmeler atlanacak")
    
    # Pipeline'ı başlat
    print("🚀 Exoplanet Detection Pipeline Başlatılıyor...")
    print("=" * 60)
    print(f"📁 CSV Dosyası: {args.csv}")
    print(f"📁 Çıktı Dizini: {args.output or 'Otomatik'}")
    print(f"⚙️  Konfigürasyon: {args.config or 'Varsayılan'}")
    print("=" * 60)
    
    try:
        # Pipeline oluştur
        pipeline = ExoplanetDetectionPipeline(config=config)
        
        # İşlemi çalıştır
        start_time = datetime.now()
        results = pipeline.process_csv(args.csv, args.output)
        end_time = datetime.now()
        
        # Sonuçları göster
        processing_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE İŞLEMİ TAMAMLANDI!")
        print("=" * 60)
        print(f"⏱️  İşlem Süresi: {processing_time:.2f} saniye")
        print(f"📁 Sonuçlar: {results['output_dir']}")
        
        if results['status'] == 'success':
            print("✅ İşlem başarılı!")
            
            # Özet istatistikler
            if 'processing_stats' in results:
                stats = results['processing_stats']
                print(f"📊 İşlenen Veri: {stats.get('input_data', {}).get('rows', 'N/A')} satır")
                print(f"🎯 Özellik Sayısı: {stats.get('feature_extraction', {}).get('basic_feature_count', 'N/A')}")
                
                # Fuzzy sonuçları
                if 'results' in results and 'fuzzy_analysis' in results['results']:
                    fuzzy = results['results']['fuzzy_analysis']
                    if 'accuracy' in fuzzy and fuzzy['accuracy'] is not None:
                        print(f"🧠 Fuzzy Accuracy: {fuzzy['accuracy']:.4f}")
                
                # PPO sonuçları
                if 'results' in results and 'ppo_analysis' in results['results']:
                    ppo = results['results']['ppo_analysis']
                    if ppo and 'total_agents' in ppo:
                        print(f"🤖 PPO Ajanları: {ppo['total_agents']} adet")
        else:
            print("❌ İşlem başarısız!")
            print(f"Hata: {results.get('error', 'Bilinmeyen hata')}")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()