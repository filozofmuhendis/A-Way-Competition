import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import torch
import warnings
warnings.filterwarnings('ignore')

from fuzzy_logic_system import ExoplanetFuzzySystem
from rl_environment import MultiAgentEnvironment, AgentType
from ppo_agent import PPOTrainer
from enhanced_ppo_agents import EnhancedMultiAgentPPOTrainer, FuzzyOutputType

class IntegratedExoplanetSystem:
    """
    Bulanık mantık + Desteklemeli öğrenme entegre sistemi
    """
    
    def __init__(self, data_path='feature_extracted__k2_dataset_clean.csv', use_enhanced_system=True):
        self.data_path = data_path
        self.use_enhanced_system = use_enhanced_system
        self.df = None
        self.fuzzy_system = None
        self.multi_env = None
        self.ppo_trainer = None
        self.enhanced_ppo_trainer = None
        
        # Sistem bileşenlerini başlat
        self.initialize_system()
    
    def initialize_system(self):
        """Sistem bileşenlerini başlat"""
        print("Entegre Exoplanet Detection Sistemi Başlatılıyor...")
        print("=" * 60)
        
        # 1. Veriyi yükle
        print("1. Veri yükleniyor...")
        self.df = pd.read_csv(self.data_path)
        self.data = self.df  # Test uyumluluğu için data özelliği ekle
        print(f"   Toplam örnek sayısı: {len(self.df)}")
        print(f"   Pozitif örnekler: {sum(self.df['label'])}")
        print(f"   Negatif örnekler: {len(self.df) - sum(self.df['label'])}")
        
        # 2. Bulanık mantık sistemini oluştur
        print("\n2. Bulanık mantık sistemi oluşturuluyor...")
        self.fuzzy_system = ExoplanetFuzzySystem()
        
        # 3. Multi-agent ortamını oluştur
        print("\n3. Multi-agent desteklemeli öğrenme ortamı oluşturuluyor...")
        self.multi_env = MultiAgentEnvironment(self.fuzzy_system, self.df)
        
        # Ortam istatistiklerini göster
        stats = self.multi_env.get_environment_stats()
        for agent_type, stat in stats.items():
            print(f"   {agent_type.upper()} Ajan - Veri boyutu: {stat['data_size']}, "
                  f"Gözlem boyutu: {stat['observation_space']}")
        
        # 4. PPO trainer'ı oluştur
        print("\n4. PPO trainer oluşturuluyor...")
        self.ppo_trainer = PPOTrainer(self.multi_env, max_episodes=500)
        
        # 5. Enhanced PPO trainer'ı oluştur (eğer enhanced mode aktifse)
        if self.use_enhanced_system:
            print("\n5. Geliştirilmiş PPO trainer oluşturuluyor...")
            self.enhanced_ppo_trainer = EnhancedMultiAgentPPOTrainer(
                self.multi_env, 
                self.fuzzy_system, 
                max_episodes=500
            )
            print("   Özelleştirilmiş ajanlar oluşturuldu:")
            for fuzzy_output in FuzzyOutputType:
                for agent_type in AgentType:
                    agent_key = f"{fuzzy_output.value}_{agent_type.value}"
                    print(f"     - {agent_key}")
        
        print("\nSistem başarıyla başlatıldı!")
    
    def run_fuzzy_analysis(self):
        """Bulanık mantık analizi"""
        print("\nBulanık Mantık Analizi...")
        print("-" * 40)
        
        # Özellikleri çıkar
        features = self.fuzzy_system.extract_features_from_data(self.df)
        normalized_features, scaler = self.fuzzy_system.normalize_features(features)
        
        # Bulanık mantık tahminleri
        fuzzy_predictions = self.fuzzy_system.predict(normalized_features)
        
        # Performans değerlendirmesi
        binary_predictions = (fuzzy_predictions / 100.0 > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(self.df['label'], binary_predictions)
        precision = precision_score(self.df['label'], binary_predictions)
        recall = recall_score(self.df['label'], binary_predictions)
        f1 = f1_score(self.df['label'], binary_predictions)
        
        print(f"Bulanık Mantık Performansı:")
        print(f"  Doğruluk: {accuracy:.4f}")
        print(f"  Kesinlik: {precision:.4f}")
        print(f"  Duyarlılık: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Sonuçları kaydet
        self.fuzzy_results = {
            'predictions': fuzzy_predictions,
            'binary_predictions': binary_predictions,
            'features': normalized_features,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return self.fuzzy_results
    
    def train_rl_agents(self, episodes_per_agent=200, use_enhanced=None):
        """Desteklemeli öğrenme ajanlarını eğit"""
        if use_enhanced is None:
            use_enhanced = self.use_enhanced_system
            
        if use_enhanced and self.enhanced_ppo_trainer:
            print(f"\nGeliştirilmiş Desteklemeli Öğrenme Ajanları Eğitiliyor ({episodes_per_agent} episode)...")
            print("-" * 60)
            
            # Geliştirilmiş ajanları eğit
            enhanced_results = self.enhanced_ppo_trainer.train_specialized_agents(episodes_per_agent)
            
            # Sonuçları kaydet
            self.enhanced_training_results = enhanced_results
            
            print("\nGeliştirilmiş Eğitim Sonuçları:")
            for agent_key, results in enhanced_results.items():
                print(f"  {agent_key}: Final Avg Reward = {results['final_avg_reward']:.4f}")
            
            return enhanced_results
        else:
            print(f"\nStandart Desteklemeli Öğrenme Ajanları Eğitiliyor ({episodes_per_agent} episode)...")
            print("-" * 60)
            
            # Tüm ajanları eğit
            training_results = self.ppo_trainer.train_all_agents(episodes_per_agent)
            
            # Eğitim sonuçlarını kaydet
            self.training_results = training_results
            
            return training_results
    
    def evaluate_system(self, test_episodes=100):
        """Entegre sistemi değerlendir"""
        print(f"\nSistem Değerlendirmesi ({test_episodes} test episode)...")
        print("-" * 50)
        
        # RL ajanlarını değerlendir
        rl_results = self.ppo_trainer.evaluate_agents(test_episodes)
        
        # Sonuçları karşılaştır
        print("\nPerformans Karşılaştırması:")
        print("=" * 50)
        
        # Bulanık mantık sonuçları
        if hasattr(self, 'fuzzy_results'):
            print(f"Bulanık Mantık Sistemi:")
            print(f"  Doğruluk: {self.fuzzy_results['accuracy']:.4f}")
            print(f"  F1-Score: {self.fuzzy_results['f1_score']:.4f}")
        
        # RL ajanları sonuçları
        print(f"\nDesteklemeli Öğrenme Ajanları:")
        for agent_type, results in rl_results.items():
            agent_name = agent_type.name if hasattr(agent_type, 'name') else str(agent_type)
            print(f"  {agent_name.upper()} Ajan:")
            print(f"    Doğruluk: {results['avg_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"    Ortalama Ödül: {results['avg_reward']:.4f} ± {results['std_reward']:.4f}")
        
        self.evaluation_results = rl_results
        return rl_results
    
    def create_ensemble_predictions(self, test_episodes=50):
        """Ensemble tahminleri oluştur"""
        print("\nEnsemble Tahminleri Oluşturuluyor...")
        print("-" * 40)
        
        # Bulanık mantık tahminleri
        if not hasattr(self, 'fuzzy_results'):
            self.run_fuzzy_analysis()
        
        fuzzy_probs = self.fuzzy_results['predictions'] / 100.0
        
        # RL ajanları tahminleri
        rl_predictions = {}
        
        for agent_type, agent in self.ppo_trainer.agents.items():
            env = self.multi_env.get_environment(agent_type)
            predictions = []
            
            # Test verisi üzerinde tahmin yap
            for idx in range(len(env.filtered_data)):
                env.current_step = idx
                state = env._get_observation()
                
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, _ = agent.policy(state_tensor)
                    prob = action_probs[0, 1].item()  # Exoplanet olasılığı
                    predictions.append(prob)
            
            rl_predictions[agent_type] = np.array(predictions)
        
        # Ensemble oluştur (ağırlıklı ortalama)
        weights = {
            AgentType.SIMPLE: 0.2,
            AgentType.INTERMEDIATE: 0.4,
            AgentType.COMPLEX: 0.4
        }
        
        # RL ensemble
        rl_ensemble = np.zeros(len(self.df))
        for agent_type, preds in rl_predictions.items():
            # Veri boyutlarını eşitle
            if len(preds) != len(self.df):
                # Interpolation veya padding
                if len(preds) < len(self.df):
                    preds = np.pad(preds, (0, len(self.df) - len(preds)), 'edge')
                else:
                    preds = preds[:len(self.df)]
            
            rl_ensemble += weights[agent_type] * preds
        
        # Final ensemble (Bulanık + RL)
        final_ensemble = 0.3 * fuzzy_probs + 0.7 * rl_ensemble
        
        # Performans değerlendirmesi
        binary_ensemble = (final_ensemble > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, classification_report
        
        ensemble_accuracy = accuracy_score(self.df['label'], binary_ensemble)
        
        print(f"Ensemble Performansı:")
        print(f"  Doğruluk: {ensemble_accuracy:.4f}")
        print(f"\nDetaylı Rapor:")
        print(classification_report(self.df['label'], binary_ensemble))
        
        self.ensemble_results = {
            'fuzzy_predictions': fuzzy_probs,
            'rl_predictions': rl_predictions,
            'rl_ensemble': rl_ensemble,
            'final_ensemble': final_ensemble,
            'binary_predictions': binary_ensemble,
            'accuracy': ensemble_accuracy
        }
        
        return self.ensemble_results
    
    def visualize_results(self):
        """Sonuçları görselleştir"""
        print("\nSonuçlar görselleştiriliyor...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Bulanık mantık üyelik fonksiyonları
        plt.subplot(3, 4, 1)
        self.fuzzy_system.snr_transit.view()
        plt.title('SNR Transit Üyelik Fonksiyonları')
        
        plt.subplot(3, 4, 2)
        self.fuzzy_system.beta_factor.view()
        plt.title('Beta Faktörü Üyelik Fonksiyonları')
        
        plt.subplot(3, 4, 3)
        self.fuzzy_system.exoplanet_probability.view()
        plt.title('Exoplanet Olasılığı Üyelik Fonksiyonları')
        
        # 2. Eğitim eğrileri
        if hasattr(self, 'training_results'):
            colors = ['blue', 'green', 'red']
            agent_types = [AgentType.SIMPLE, AgentType.INTERMEDIATE, AgentType.COMPLEX]
            
            plt.subplot(3, 4, 4)
            for i, agent_type in enumerate(agent_types):
                agent = self.ppo_trainer.agents[agent_type]
                rewards = agent.training_stats['episode_rewards']
                if rewards:
                    plt.plot(rewards, color=colors[i], alpha=0.7, 
                           label=f'{agent_type.value.title()}')
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 4, 5)
            for i, agent_type in enumerate(agent_types):
                agent = self.ppo_trainer.agents[agent_type]
                accuracies = agent.training_stats['accuracies']
                if accuracies:
                    plt.plot(accuracies, color=colors[i], alpha=0.7,
                           label=f'{agent_type.value.title()}')
            plt.title('Training Accuracies')
            plt.xlabel('Episode')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
        
        # 3. Performans karşılaştırması
        if hasattr(self, 'fuzzy_results') and hasattr(self, 'evaluation_results'):
            plt.subplot(3, 4, 6)
            
            methods = ['Fuzzy Logic']
            accuracies = [self.fuzzy_results['accuracy']]
            
            for agent_type, results in self.evaluation_results.items():
                methods.append(f'RL-{agent_type.value.title()}')
                accuracies.append(results['avg_accuracy'])
            
            if hasattr(self, 'ensemble_results'):
                methods.append('Ensemble')
                accuracies.append(self.ensemble_results['accuracy'])
            
            bars = plt.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
            plt.title('Yöntem Karşılaştırması')
            plt.ylabel('Doğruluk')
            plt.xticks(rotation=45)
            
            # Değerleri çubukların üzerine yaz
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. Confusion Matrix
        if hasattr(self, 'ensemble_results'):
            plt.subplot(3, 4, 7)
            cm = confusion_matrix(self.df['label'], self.ensemble_results['binary_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Ensemble Confusion Matrix')
            plt.ylabel('Gerçek')
            plt.xlabel('Tahmin')
        
        # 5. ROC Curve
        if hasattr(self, 'ensemble_results'):
            plt.subplot(3, 4, 8)
            
            # Bulanık mantık ROC
            fpr_fuzzy, tpr_fuzzy, _ = roc_curve(self.df['label'], self.fuzzy_results['predictions']/100.0)
            auc_fuzzy = roc_auc_score(self.df['label'], self.fuzzy_results['predictions']/100.0)
            plt.plot(fpr_fuzzy, tpr_fuzzy, label=f'Fuzzy Logic (AUC={auc_fuzzy:.3f})')
            
            # Ensemble ROC
            fpr_ensemble, tpr_ensemble, _ = roc_curve(self.df['label'], self.ensemble_results['final_ensemble'])
            auc_ensemble = roc_auc_score(self.df['label'], self.ensemble_results['final_ensemble'])
            plt.plot(fpr_ensemble, tpr_ensemble, label=f'Ensemble (AUC={auc_ensemble:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True)
        
        # 6. Feature importance (bulanık mantık özellikleri)
        if hasattr(self, 'fuzzy_results'):
            plt.subplot(3, 4, 9)
            features = self.fuzzy_results['features']
            feature_importance = features.std().values
            feature_names = features.columns
            
            bars = plt.bar(feature_names, feature_importance)
            plt.title('Özellik Önem Dereceleri (Std)')
            plt.xticks(rotation=45)
            plt.ylabel('Standart Sapma')
        
        # 7. Prediction distribution
        if hasattr(self, 'ensemble_results'):
            plt.subplot(3, 4, 10)
            
            # Pozitif ve negatif örnekler için tahmin dağılımları
            pos_preds = self.ensemble_results['final_ensemble'][self.df['label'] == 1]
            neg_preds = self.ensemble_results['final_ensemble'][self.df['label'] == 0]
            
            plt.hist(neg_preds, bins=20, alpha=0.7, label='Non-Exoplanet', color='red')
            plt.hist(pos_preds, bins=20, alpha=0.7, label='Exoplanet', color='blue')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Frequency')
            plt.title('Prediction Distribution')
            plt.legend()
            plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
        
        # 8. Agent comparison
        if hasattr(self, 'evaluation_results'):
            plt.subplot(3, 4, 11)
            
            agent_names = [agent_type.value.title() for agent_type in self.evaluation_results.keys()]
            agent_accuracies = [results['avg_accuracy'] for results in self.evaluation_results.values()]
            agent_stds = [results['std_accuracy'] for results in self.evaluation_results.values()]
            
            bars = plt.bar(agent_names, agent_accuracies, yerr=agent_stds, capsize=5)
            plt.title('RL Agent Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            for bar, acc in zip(bars, agent_accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # 9. System architecture diagram (text-based)
        plt.subplot(3, 4, 12)
        plt.text(0.1, 0.8, 'ENTEGRE SİSTEM MİMARİSİ', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, '1. Ham Veri → Özellik Çıkarma', fontsize=10)
        plt.text(0.1, 0.6, '2. Bulanık Mantık Sistemi', fontsize=10)
        plt.text(0.1, 0.5, '   • 5 Özellik → Bulanık Çıktı', fontsize=8)
        plt.text(0.1, 0.4, '3. RL Ortamları (3 Ajan Tipi)', fontsize=10)
        plt.text(0.1, 0.3, '   • Simple, Intermediate, Complex', fontsize=8)
        plt.text(0.1, 0.2, '4. PPO Algoritması', fontsize=10)
        plt.text(0.1, 0.1, '5. Ensemble Tahmin', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('integrated_system_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_system(self, filepath='integrated_exoplanet_system.pkl'):
        """Sistemi kaydet"""
        import pickle
        
        save_data = {
            'fuzzy_results': getattr(self, 'fuzzy_results', None),
            'training_results': getattr(self, 'training_results', None),
            'evaluation_results': getattr(self, 'evaluation_results', None),
            'ensemble_results': getattr(self, 'ensemble_results', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # PPO ajanlarını ayrı kaydet
        if self.ppo_trainer:
            self.ppo_trainer.save_agents('integrated_ppo_agents.pkl')
        
        print(f"Sistem {filepath} dosyasına kaydedildi.")
    
    def run_complete_analysis(self, episodes_per_agent=150, test_episodes=50, compare_systems=True):
        """Tam analizi çalıştır"""
        print("ENTEGRE EXOPLANET DETECTION SİSTEMİ")
        print("=" * 60)
        print("Tam analiz başlatılıyor...")
        
        # 1. Bulanık mantık analizi
        fuzzy_results = self.run_fuzzy_analysis()
        
        # 2. RL ajanlarını eğit
        if compare_systems and self.use_enhanced_system:
            print("\n" + "="*60)
            print("SİSTEM KARŞILAŞTIRMASI YAPILIYOR")
            print("="*60)
            
            # Standart sistem
            print("\n1. STANDART SİSTEM EĞİTİMİ:")
            standard_results = self.train_rl_agents(episodes_per_agent, use_enhanced=False)
            
            # Geliştirilmiş sistem
            print("\n2. GELİŞTİRİLMİŞ SİSTEM EĞİTİMİ:")
            enhanced_results = self.train_rl_agents(episodes_per_agent, use_enhanced=True)
            
            # Karşılaştırma
            self.compare_training_results(standard_results, enhanced_results)
            
        else:
            training_results = self.train_rl_agents(episodes_per_agent)
        
        # 3. Sistemi değerlendir
        evaluation_results = self.evaluate_system(test_episodes)
        
        # 4. Ensemble tahminleri oluştur
        ensemble_results = self.create_ensemble_predictions()
        
        # 5. Sonuçları görselleştir
        self.visualize_results()
        
        # 6. Sistemi kaydet
        self.save_system()
        
        print("\n" + "=" * 60)
        print("ANALIZ TAMAMLANDI!")
        print("=" * 60)
        
        return {
            'fuzzy_results': fuzzy_results,
            'training_results': getattr(self, 'training_results', None),
            'enhanced_training_results': getattr(self, 'enhanced_training_results', None),
            'evaluation_results': evaluation_results,
            'ensemble_results': ensemble_results
        }
    
    def compare_training_results(self, standard_results, enhanced_results):
        """Standart ve geliştirilmiş sistem eğitim sonuçlarını karşılaştır"""
        print("\n" + "="*60)
        print("EĞİTİM SONUÇLARI KARŞILAŞTIRMASI")
        print("="*60)
        
        print("\nSTANDART SİSTEM SONUÇLARI:")
        print("-" * 30)
        if isinstance(standard_results, dict):
            for agent_type, results in standard_results.items():
                agent_name = agent_type.name if hasattr(agent_type, 'name') else str(agent_type)
                if 'final_avg_reward' in results:
                    print(f"  {agent_name}: {results['final_avg_reward']:.4f}")
                elif 'episode_rewards' in results:
                    final_reward = np.mean(results['episode_rewards'][-10:])
                    print(f"  {agent_name}: {final_reward:.4f}")
        
        print("\nGELİŞTİRİLMİŞ SİSTEM SONUÇLARI:")
        print("-" * 30)
        if isinstance(enhanced_results, dict):
            # Fuzzy output type'a göre grupla
            fuzzy_groups = {}
            for agent_key, results in enhanced_results.items():
                # Agent key format: "fuzzy_output_agent_type" (örn: "çok_düşük_simple")
                parts = agent_key.split('_')
                if len(parts) >= 2:
                    agent_type = parts[-1]
                    fuzzy_output = '_'.join(parts[:-1])
                else:
                    continue
                
                if fuzzy_output not in fuzzy_groups:
                    fuzzy_groups[fuzzy_output] = {}
                fuzzy_groups[fuzzy_output][agent_type] = results['final_avg_reward']
            
            for fuzzy_output, agents in fuzzy_groups.items():
                print(f"\n  {fuzzy_output.upper()} Fuzzy Çıktısı:")
                for agent_type, reward in agents.items():
                    print(f"    {agent_type}: {reward:.4f}")
        
        print("\n" + "="*60)
    
    def get_enhanced_system_stats(self):
        """Geliştirilmiş sistem istatistiklerini al"""
        if not self.enhanced_ppo_trainer:
            return None
            
        stats = {
            'total_specialized_agents': len(self.enhanced_ppo_trainer.specialized_agents),
            'fuzzy_output_types': len(FuzzyOutputType),
            'agent_types': len(AgentType),
            'agents_per_fuzzy_output': {}
        }
        
        # Her fuzzy çıktı tipi için ajan sayısı
        for fuzzy_output in FuzzyOutputType:
            count = sum(1 for key in self.enhanced_ppo_trainer.specialized_agents.keys() 
                       if key.startswith(fuzzy_output.value))
            stats['agents_per_fuzzy_output'][fuzzy_output.value] = count
        
        return stats

def main():
    """Ana fonksiyon"""
    # Entegre sistemi oluştur ve çalıştır
    system = IntegratedExoplanetSystem()
    
    # Tam analizi çalıştır
    results = system.run_complete_analysis(
        episodes_per_agent=200,  # Her ajan için episode sayısı
        test_episodes=50         # Test episode sayısı
    )
    
    return system, results

if __name__ == "__main__":
    system, results = main()