"""
Exoplanet Detection Pipeline
============================

Bu modül, geliştirdiğimiz tüm sistemleri entegre eden kapsamlı bir 
exoplanet detection pipeline'ı sağlar.

Özellikler:
- CSV dosyalarını okuma ve işleme
- Özellik çıkarımı (feature extraction)
- Geliştirilmiş Fuzzy Logic analizi
- Özelleştirilmiş PPO ajanları ile eğitim
- Sonuçları CSV formatında kaydetme
- Görselleştirme ve raporlama

Kullanım:
    pipeline = ExoplanetDetectionPipeline()
    results = pipeline.process_csv('data.csv')
    pipeline.save_results(results, 'output.csv')
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Kendi modüllerimizi import et
from fuzzy_logic_system import ExoplanetFuzzySystem
from enhanced_fuzzy_rules import EnhancedFuzzyRuleGenerator
from enhanced_ppo_agents import EnhancedMultiAgentPPOTrainer, FuzzyOutputType, AgentType
from integrated_system import IntegratedExoplanetSystem
from preprocess import preprocess_data
from audio_inspired_preprocessor import AudioInspiredPreprocessor


class ExoplanetDetectionPipeline:
    """
    Kapsamlı Exoplanet Detection Pipeline
    
    Bu sınıf, tüm geliştirdiğimiz sistemleri entegre ederek
    CSV dosyalarından exoplanet tespiti yapan bir pipeline sağlar.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Pipeline'ı başlat
        
        Args:
            config: Pipeline konfigürasyonu
        """
        self.config = config or self._get_default_config()
        self.results_history = []
        self.processing_stats = {}
        
        # Sistemleri başlat
        self._initialize_systems()
        
        print("🚀 Exoplanet Detection Pipeline Başlatıldı!")
        print("=" * 60)
        self._print_system_info()
    
    def _get_default_config(self) -> Dict:
        """Varsayılan konfigürasyon"""
        return {
            'fuzzy_system': {
                'enable_enhanced_rules': True,
                'rule_count_target': 200
            },
            'ppo_system': {
                'enable_specialized_agents': True,
                'episodes_per_combination': 50,
                'enable_training': True
            },
            'preprocessing': {
                'enable_audio_features': True,
                'normalize_features': True,
                'handle_missing_values': True
            },
            'output': {
                'save_intermediate_results': True,
                'create_visualizations': True,
                'detailed_reports': True
            }
        }
    
    def _initialize_systems(self):
        """Tüm sistemleri başlat"""
        print("🔧 Sistemler başlatılıyor...")
        
        # Fuzzy Logic Sistemi
        self.fuzzy_system = ExoplanetFuzzySystem()
        print("✓ Fuzzy Logic Sistemi hazır")
        
        # Enhanced Fuzzy Rules
        if self.config.get('fuzzy_system', {}).get('enable_enhanced_rules', True):
            self.enhanced_rules = EnhancedFuzzyRuleGenerator()
            print("✓ Geliştirilmiş Fuzzy Kuralları hazır")
        
        # PPO Sistemi - Gerekli parametreler olmadan başlatmayacağız
        self.ppo_trainer = None
        if self.config.get('ppo_system', {}).get('enable_specialized_agents', True):
            print("✓ PPO sistemi hazır (gerektiğinde başlatılacak)")
        
        # Audio Preprocessor
        if self.config.get('preprocessing', {}).get('enable_audio_features', True):
            self.audio_processor = AudioInspiredPreprocessor()
            print("✓ Audio-Inspired Preprocessor hazır")
        
        # Entegre Sistem
        self.integrated_system = None  # Gerektiğinde başlatılacak
        
        print("✅ Tüm sistemler başarıyla başlatıldı!\n")
    
    def _print_system_info(self):
        """Sistem bilgilerini yazdır"""
        print("📊 SİSTEM BİLGİLERİ:")
        print(f"  • Fuzzy Logic: {'✓ Aktif' if self.fuzzy_system else '✗ Pasif'}")
        print(f"  • Enhanced Rules: {'✓ Aktif' if hasattr(self, 'enhanced_rules') else '✗ Pasif'}")
        print(f"  • PPO Agents: {'✓ Aktif' if hasattr(self, 'ppo_trainer') else '✗ Pasif'}")
        print(f"  • Audio Features: {'✓ Aktif' if hasattr(self, 'audio_processor') else '✗ Pasif'}")
        print()
    
    def process_csv(self, csv_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        CSV dosyasını işle ve exoplanet tespiti yap
        
        Args:
            csv_path: İşlenecek CSV dosyasının yolu
            output_dir: Çıktı dizini (varsayılan: 'pipeline_results')
            
        Returns:
            İşlem sonuçları dictionary'si
        """
        print(f"📁 CSV İşleme Başlatıldı: {csv_path}")
        print("=" * 60)
        
        # Çıktı dizinini hazırla
        if output_dir is None:
            output_dir = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. CSV'yi oku
            print("📖 1. CSV Dosyası Okunuyor...")
            data = self._load_csv(csv_path)
            
            # 2. Veri ön işleme
            print("🔧 2. Veri Ön İşleme...")
            processed_data = self._preprocess_data(data)
            
            # 3. Özellik çıkarımı
            print("🎯 3. Özellik Çıkarımı...")
            features = self._extract_features(processed_data)
            
            # 4. Fuzzy Logic analizi
            print("🧠 4. Fuzzy Logic Analizi...")
            fuzzy_results = self._run_fuzzy_analysis(features)
            
            # 5. PPO analizi (eğer etkinse)
            ppo_results = None
            if self.config['ppo_system']['enable_training']:
                print("🤖 5. PPO Ajanları Eğitimi...")
                ppo_results = self._run_ppo_analysis(features)
            
            # 6. Sonuçları birleştir
            print("📊 6. Sonuçlar Birleştiriliyor...")
            combined_results = self._combine_results(
                data, features, fuzzy_results, ppo_results
            )
            
            # 7. Sonuçları kaydet
            print("💾 7. Sonuçlar Kaydediliyor...")
            self._save_results(combined_results, output_dir)
            
            # 8. Görselleştirme
            if self.config.get('output', {}).get('create_visualizations', True):
                print("📈 8. Görselleştirmeler Oluşturuluyor...")
                self._create_visualizations(combined_results, output_dir)
            
            # 9. Rapor oluştur
            if self.config.get('output', {}).get('detailed_reports', True):
                print("📋 9. Detaylı Rapor Oluşturuluyor...")
                self._create_report(combined_results, output_dir)
            
            print("\n🎉 Pipeline İşlemi Başarıyla Tamamlandı!")
            print(f"📁 Sonuçlar: {output_dir}")
            
            return {
                'status': 'success',
                'output_dir': output_dir,
                'results': combined_results,
                'processing_stats': self.processing_stats
            }
            
        except Exception as e:
            print(f"❌ Pipeline Hatası: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': str(e),
                'output_dir': output_dir
            }
    
    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """CSV dosyasını yükle"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        
        data = pd.read_csv(csv_path)
        print(f"  ✓ {len(data)} satır, {len(data.columns)} sütun yüklendi")
        
        # Temel istatistikler
        self.processing_stats['input_data'] = {
            'rows': len(data),
            'columns': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Veriyi ön işle"""
        processed_data = data.copy()
        
        # Eksik değerleri işle
        if self.config.get('preprocessing', {}).get('handle_missing_values', True):
            # Sayısal sütunlar için ortalama ile doldur
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_columns] = processed_data[numeric_columns].fillna(
                processed_data[numeric_columns].mean()
            )
            print(f"  ✓ {len(numeric_columns)} sayısal sütunda eksik değerler dolduruldu")
        
        # Normalizasyon
        if self.config.get('preprocessing', {}).get('normalize_features', True):
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            # Label sütununu normalizasyondan hariç tut
            feature_columns = [col for col in numeric_columns if col.lower() not in ['label', 'target', 'class']]
            
            if feature_columns:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                processed_data[feature_columns] = scaler.fit_transform(processed_data[feature_columns])
                print(f"  ✓ {len(feature_columns)} özellik normalize edildi")
        
        self.processing_stats['preprocessing'] = {
            'normalized_features': len(feature_columns) if 'feature_columns' in locals() else 0,
            'final_shape': processed_data.shape
        }
        
        return processed_data
    
    def _extract_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Özellik çıkarımı yap"""
        features = {}
        
        # Temel özellikler
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col.lower() not in ['label', 'target', 'class']]
        
        features['basic_features'] = data[feature_columns].values
        features['feature_names'] = list(feature_columns)
        
        # Audio-inspired özellikler (eğer etkinse)
        if hasattr(self, 'audio_processor'):
            try:
                # İlk birkaç özelliği kullanarak audio features çıkar
                sample_data = data[feature_columns].iloc[:min(100, len(data))]
                audio_features = []
                
                for idx, row in sample_data.iterrows():
                    try:
                        audio_feat = self.audio_processor.extract_audio_features(row.values)
                        audio_features.append(audio_feat)
                    except:
                        # Hata durumunda sıfır vektör ekle
                        audio_features.append(np.zeros(10))
                
                features['audio_features'] = np.array(audio_features)
                print(f"  ✓ {len(audio_features)} örnek için audio özellikler çıkarıldı")
                
            except Exception as e:
                print(f"  ⚠️ Audio özellik çıkarımında hata: {e}")
                features['audio_features'] = None
        
        # Label bilgisi (varsa)
        label_columns = [col for col in data.columns if col.lower() in ['label', 'target', 'class']]
        if label_columns:
            features['labels'] = data[label_columns[0]].values
            print(f"  ✓ Label sütunu bulundu: {label_columns[0]}")
        else:
            features['labels'] = None
            print("  ℹ️ Label sütunu bulunamadı")
        
        self.processing_stats['feature_extraction'] = {
            'basic_feature_count': len(feature_columns),
            'audio_feature_count': features['audio_features'].shape[1] if features['audio_features'] is not None else 0,
            'sample_count': len(features['basic_features'])
        }
        
        return features
    
    def _run_fuzzy_analysis(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fuzzy Logic analizi çalıştır"""
        try:
            # Temel özellikleri DataFrame'e çevir
            feature_df = pd.DataFrame(
                features['basic_features'], 
                columns=features['feature_names']
            )
            
            # Fuzzy predictions
            predictions = self.fuzzy_system.predict(feature_df)
            probabilities = predictions / 100.0  # 0-1 aralığına normalize et
            
            # Enhanced rules (eğer etkinse)
            enhanced_rules = None
            if hasattr(self, 'enhanced_rules'):
                try:
                    enhanced_rules = self.enhanced_rules.generate_comprehensive_rules()
                    print(f"  ✓ {len(enhanced_rules)} geliştirilmiş kural oluşturuldu")
                except Exception as e:
                    print(f"  ⚠️ Enhanced rules hatası: {e}")
            
            # Sonuçları hesapla
            binary_predictions = (probabilities > 0.5).astype(int)
            
            # Accuracy hesapla (eğer label varsa)
            accuracy = None
            if features['labels'] is not None:
                accuracy = np.mean(binary_predictions == features['labels'])
                print(f"  ✓ Fuzzy Logic Accuracy: {accuracy:.4f}")
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'binary_predictions': binary_predictions,
                'accuracy': accuracy,
                'enhanced_rules': enhanced_rules,
                'rule_count': len(enhanced_rules) if enhanced_rules else 0
            }
            
        except Exception as e:
            print(f"  ❌ Fuzzy analiz hatası: {e}")
            return {'error': str(e)}
    
    def _run_ppo_analysis(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """PPO analizi çalıştır"""
        try:
            # Entegre sistemi başlat (eğer henüz başlatılmamışsa)
            if self.integrated_system is None:
                # Geçici bir CSV dosyası oluştur
                temp_csv = 'temp_pipeline_data.csv'
                temp_df = pd.DataFrame(features['basic_features'], columns=features['feature_names'])
                if features['labels'] is not None:
                    temp_df['label'] = features['labels']
                temp_df.to_csv(temp_csv, index=False)
                
                self.integrated_system = IntegratedExoplanetSystem(
                    data_path=temp_csv,
                    use_enhanced_system=True
                )
            
            # PPO eğitimi çalıştır
            episodes = self.config['ppo_system']['episodes_per_combination']
            training_results = self.ppo_trainer.train_specialized_agents(
                episodes_per_combination=episodes
            )
            
            print(f"  ✓ {len(training_results)} PPO ajanı eğitildi")
            
            # En iyi performans gösteren ajanları bul
            best_agents = {}
            for agent_key, results in training_results.items():
                reward = results.get('final_avg_reward', 0)
                fuzzy_type = agent_key.split('_')[0] + '_' + agent_key.split('_')[1]  # fuzzy output type
                
                if fuzzy_type not in best_agents or reward > best_agents[fuzzy_type]['reward']:
                    best_agents[fuzzy_type] = {
                        'agent_key': agent_key,
                        'reward': reward,
                        'results': results
                    }
            
            return {
                'training_results': training_results,
                'best_agents': best_agents,
                'total_agents': len(training_results),
                'avg_reward': np.mean([r.get('final_avg_reward', 0) for r in training_results.values()])
            }
            
        except Exception as e:
            print(f"  ❌ PPO analiz hatası: {e}")
            return {'error': str(e)}
    
    def _combine_results(self, original_data: pd.DataFrame, features: Dict[str, Any], 
                        fuzzy_results: Dict[str, Any], ppo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Tüm sonuçları birleştir"""
        
        combined = {
            'input_data': {
                'shape': original_data.shape,
                'columns': original_data.columns.tolist(),
                'sample_data': original_data.head().to_dict()
            },
            'features': {
                'basic_feature_count': len(features['feature_names']),
                'feature_names': features['feature_names'],
                'has_labels': features['labels'] is not None,
                'sample_count': len(features['basic_features'])
            },
            'fuzzy_analysis': fuzzy_results,
            'ppo_analysis': ppo_results,
            'processing_stats': self.processing_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Detaylı sonuçlar tablosu oluştur
        results_df = pd.DataFrame()
        results_df['sample_id'] = range(len(features['basic_features']))
        
        # Fuzzy sonuçları ekle
        if 'probabilities' in fuzzy_results:
            results_df['fuzzy_probability'] = fuzzy_results['probabilities']
            results_df['fuzzy_prediction'] = fuzzy_results['binary_predictions']
        
        # Gerçek labelları ekle (varsa)
        if features['labels'] is not None:
            results_df['true_label'] = features['labels']
            results_df['fuzzy_correct'] = (results_df['fuzzy_prediction'] == results_df['true_label']).astype(int)
        
        combined['detailed_results'] = results_df
        
        return combined
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Sonuçları kaydet"""
        
        # Ana sonuçlar JSON olarak
        with open(os.path.join(output_dir, 'pipeline_results.json'), 'w', encoding='utf-8') as f:
            # JSON serializable hale getir
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # Detaylı sonuçlar CSV olarak
        if 'detailed_results' in results:
            results['detailed_results'].to_csv(
                os.path.join(output_dir, 'detailed_predictions.csv'), 
                index=False
            )
        
        # Fuzzy kuralları (varsa)
        if 'fuzzy_analysis' in results and 'enhanced_rules' in results['fuzzy_analysis']:
            rules = results['fuzzy_analysis']['enhanced_rules']
            if rules:
                rules_df = pd.DataFrame(rules)
                rules_df.to_csv(
                    os.path.join(output_dir, 'fuzzy_rules.csv'), 
                    index=False
                )
        
        print(f"  ✓ Sonuçlar kaydedildi: {output_dir}")
    
    def _make_json_serializable(self, obj):
        """Objeyi JSON serializable hale getir"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj) if not isinstance(obj, (pd.DataFrame, pd.Series)) else False:
            return None
        else:
            return obj
    
    def _create_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Görselleştirmeler oluştur"""
        try:
            plt.style.use('default')
            
            # 1. Fuzzy Probability Dağılımı
            if 'fuzzy_analysis' in results and 'probabilities' in results['fuzzy_analysis']:
                plt.figure(figsize=(12, 8))
                
                # Subplot 1: Probability histogram
                plt.subplot(2, 2, 1)
                probabilities = results['fuzzy_analysis']['probabilities']
                plt.hist(probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title('Fuzzy Logic Probability Dağılımı')
                plt.xlabel('Exoplanet Probability')
                plt.ylabel('Frekans')
                plt.grid(True, alpha=0.3)
                
                # Subplot 2: Prediction distribution
                plt.subplot(2, 2, 2)
                predictions = results['fuzzy_analysis']['binary_predictions']
                unique, counts = np.unique(predictions, return_counts=True)
                plt.bar(unique, counts, color=['lightcoral', 'lightgreen'])
                plt.title('Fuzzy Logic Tahmin Dağılımı')
                plt.xlabel('Tahmin (0: No Exoplanet, 1: Exoplanet)')
                plt.ylabel('Sayı')
                plt.xticks(unique)
                
                # Subplot 3: Confusion Matrix (eğer label varsa)
                if 'detailed_results' in results and 'true_label' in results['detailed_results'].columns:
                    plt.subplot(2, 2, 3)
                    df = results['detailed_results']
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(df['true_label'], df['fuzzy_prediction'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                
                # Subplot 4: PPO Results (eğer varsa)
                if 'ppo_analysis' in results and 'best_agents' in results['ppo_analysis']:
                    plt.subplot(2, 2, 4)
                    best_agents = results['ppo_analysis']['best_agents']
                    agent_names = list(best_agents.keys())
                    rewards = [best_agents[name]['reward'] for name in agent_names]
                    
                    plt.bar(range(len(agent_names)), rewards, color='gold')
                    plt.title('En İyi PPO Ajanları - Final Rewards')
                    plt.xlabel('Fuzzy Output Type')
                    plt.ylabel('Final Reward')
                    plt.xticks(range(len(agent_names)), agent_names, rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pipeline_visualizations.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print("  ✓ Görselleştirmeler oluşturuldu")
                
        except Exception as e:
            print(f"  ⚠️ Görselleştirme hatası: {e}")
    
    def _create_report(self, results: Dict[str, Any], output_dir: str):
        """Detaylı rapor oluştur"""
        try:
            report_path = os.path.join(output_dir, 'pipeline_report.md')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Exoplanet Detection Pipeline Raporu\n\n")
                f.write(f"**Tarih:** {results['timestamp']}\n\n")
                
                # Veri Özeti
                f.write("## 📊 Veri Özeti\n\n")
                input_data = results['input_data']
                f.write(f"- **Toplam Örnek:** {input_data['shape'][0]}\n")
                f.write(f"- **Özellik Sayısı:** {input_data['shape'][1]}\n")
                f.write(f"- **Sütunlar:** {', '.join(input_data['columns'])}\n\n")
                
                # Fuzzy Logic Sonuçları
                f.write("## 🧠 Fuzzy Logic Analizi\n\n")
                if 'fuzzy_analysis' in results and 'accuracy' in results['fuzzy_analysis']:
                    fuzzy = results['fuzzy_analysis']
                    if fuzzy['accuracy'] is not None:
                        f.write(f"- **Accuracy:** {fuzzy['accuracy']:.4f}\n")
                    f.write(f"- **Kural Sayısı:** {fuzzy.get('rule_count', 'N/A')}\n")
                    f.write(f"- **Ortalama Probability:** {np.mean(fuzzy['probabilities']):.4f}\n\n")
                
                # PPO Sonuçları
                if 'ppo_analysis' in results and 'total_agents' in results['ppo_analysis']:
                    f.write("## 🤖 PPO Ajanları Analizi\n\n")
                    ppo = results['ppo_analysis']
                    f.write(f"- **Toplam Ajan:** {ppo['total_agents']}\n")
                    f.write(f"- **Ortalama Reward:** {ppo['avg_reward']:.4f}\n\n")
                    
                    if 'best_agents' in ppo:
                        f.write("### En İyi Ajanlar\n\n")
                        for fuzzy_type, agent_info in ppo['best_agents'].items():
                            f.write(f"- **{fuzzy_type}:** {agent_info['agent_key']} (Reward: {agent_info['reward']:.4f})\n")
                        f.write("\n")
                
                # İşlem İstatistikleri
                f.write("## 📈 İşlem İstatistikleri\n\n")
                stats = results['processing_stats']
                for category, category_stats in stats.items():
                    f.write(f"### {category.title()}\n")
                    for key, value in category_stats.items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")
                
                f.write("---\n")
                f.write("*Bu rapor Exoplanet Detection Pipeline tarafından otomatik olarak oluşturulmuştur.*\n")
            
            print(f"  ✓ Detaylı rapor oluşturuldu: {report_path}")
            
        except Exception as e:
            print(f"  ⚠️ Rapor oluşturma hatası: {e}")


def main():
    """Ana fonksiyon - Pipeline'ı test et"""
    # Örnek kullanım
    pipeline = ExoplanetDetectionPipeline()
    
    # K2 dataset ile test
    csv_path = 'feature_extracted__k2_dataset_clean.csv'
    if os.path.exists(csv_path):
        results = pipeline.process_csv(csv_path)
        print(f"\n🎉 Pipeline testi tamamlandı!")
        print(f"📁 Sonuçlar: {results['output_dir']}")
    else:
        print(f"❌ Test dosyası bulunamadı: {csv_path}")


if __name__ == "__main__":
    main()