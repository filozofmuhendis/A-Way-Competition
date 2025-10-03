import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt

class AgentType(Enum):
    """Ajan tipleri"""
    SIMPLE = "simple"      # Basit veriler için
    INTERMEDIATE = "intermediate"  # Orta karmaşıklık
    COMPLEX = "complex"    # Karmaşık veriler için

class ExoplanetRLEnvironment(gym.Env):
    """
    Exoplanet detection için desteklemeli öğrenme ortamı
    Bulanık mantık çıktısını kullanarak üç farklı ajan tipini destekler
    """
    
    def __init__(self, fuzzy_system, data_df, agent_type=AgentType.INTERMEDIATE):
        super(ExoplanetRLEnvironment, self).__init__()
        
        self.fuzzy_system = fuzzy_system
        self.data_df = data_df.copy()
        self.agent_type = agent_type
        
        # Veri karmaşıklığına göre filtreleme
        self.filtered_data = self._filter_data_by_complexity()
        
        # Durum ve aksiyon uzaylarını tanımla
        self._setup_spaces()
        
        # Ortam durumunu başlat
        self.reset()
        
    def _filter_data_by_complexity(self):
        """Ajan tipine göre veriyi filtrele"""
        if self.agent_type == AgentType.SIMPLE:
            # Basit ajan: Sadece açık exoplanet sinyalleri
            mask = (self.data_df['bls_depth'] > self.data_df['bls_depth'].quantile(0.7)) | \
                   (self.data_df['label'] == 0)
            return self.data_df[mask].reset_index(drop=True)
            
        elif self.agent_type == AgentType.INTERMEDIATE:
            # Orta ajan: Tüm veri
            return self.data_df.reset_index(drop=True)
            
        else:  # COMPLEX
            # Karmaşık ajan: Zor örneklere odaklan
            mask = (self.data_df['bls_depth'] < self.data_df['bls_depth'].quantile(0.5)) & \
                   (self.data_df['pdcsap_flux_std'] > self.data_df['pdcsap_flux_std'].quantile(0.5))
            complex_data = self.data_df[mask]
            if len(complex_data) < 10:  # Minimum veri garantisi
                return self.data_df.reset_index(drop=True)
            return complex_data.reset_index(drop=True)
    
    def _setup_spaces(self):
        """Durum ve aksiyon uzaylarını tanımla"""
        
        # Durum uzayı: Bulanık mantık çıktısı + ham özellikler
        if self.agent_type == AgentType.SIMPLE:
            # Basit ajan: Sadece bulanık çıktı ve temel özellikler
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, 
                shape=(3,),  # fuzzy_output, bls_depth_norm, bls_period_norm
                dtype=np.float32
            )
        elif self.agent_type == AgentType.INTERMEDIATE:
            # Orta ajan: Bulanık çıktı + 5 özellik
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(6,),  # fuzzy_output + 5 normalized features
                dtype=np.float32
            )
        else:  # COMPLEX
            # Karmaşık ajan: Tüm özellikler + istatistikler
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(10,),  # fuzzy_output + 5 features + 4 additional stats
                dtype=np.float32
            )
        
        # Aksiyon uzayı: [exoplanet_yok, exoplanet_var]
        self.action_space = spaces.Discrete(2)
        
    def reset(self):
        """Ortamı sıfırla"""
        self.current_step = 0
        self.total_reward = 0
        self.correct_predictions = 0
        self.episode_length = min(len(self.filtered_data), 100)  # Maksimum 100 adım
        
        # İlk gözlemi döndür
        return self._get_observation()
    
    def _get_observation(self):
        """Mevcut durum için gözlem vektörünü oluştur"""
        if self.current_step >= len(self.filtered_data):
            self.current_step = 0
            
        current_data = self.filtered_data.iloc[self.current_step]
        
        # Bulanık mantık çıktısını al
        features = self.fuzzy_system.extract_features_from_data(
            pd.DataFrame([current_data])
        )
        normalized_features, _ = self.fuzzy_system.normalize_features(features)
        fuzzy_output = self.fuzzy_system.predict(normalized_features)[0] / 100.0
        
        if self.agent_type == AgentType.SIMPLE:
            # Basit ajan için minimal özellikler
            obs = np.array([
                fuzzy_output,
                min(current_data['bls_depth'] / 10000.0, 1.0),  # Normalize depth
                min(current_data['bls_period'] / 50.0, 1.0)     # Normalize period
            ], dtype=np.float32)
            
        elif self.agent_type == AgentType.INTERMEDIATE:
            # Orta ajan için 5 özellik + bulanık çıktı
            obs = np.array([
                fuzzy_output,
                normalized_features.iloc[0]['snr_transit'] / 100.0,
                normalized_features.iloc[0]['beta_factor'] / 100.0,
                normalized_features.iloc[0]['odd_even_diff'] / 100.0,
                normalized_features.iloc[0]['duty_cycle'] / 100.0,
                normalized_features.iloc[0]['depth_consistency'] / 100.0
            ], dtype=np.float32)
            
        else:  # COMPLEX
            # Karmaşık ajan için ek istatistikler
            flux_cv = current_data['pdcsap_flux_std'] / (current_data['pdcsap_flux_mean'] + 1e-8)
            depth_ratio = current_data['bls_depth'] / (current_data['pdcsap_flux_mean'] + 1e-8)
            
            obs = np.array([
                fuzzy_output,
                normalized_features.iloc[0]['snr_transit'] / 100.0,
                normalized_features.iloc[0]['beta_factor'] / 100.0,
                normalized_features.iloc[0]['odd_even_diff'] / 100.0,
                normalized_features.iloc[0]['duty_cycle'] / 100.0,
                normalized_features.iloc[0]['depth_consistency'] / 100.0,
                min(flux_cv, 1.0),  # Coefficient of variation
                min(depth_ratio, 1.0),  # Depth ratio
                min(current_data['bls_duration'], 1.0),  # Duration
                min(current_data['pdcsap_flux_err_mean'] / 100.0, 1.0)  # Error mean
            ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Bir adım ilerle"""
        if self.current_step >= len(self.filtered_data):
            return self._get_observation(), 0, True, {}
            
        current_data = self.filtered_data.iloc[self.current_step]
        true_label = int(current_data['label'])
        
        # Ödül hesapla
        reward = self._calculate_reward(action, true_label, current_data)
        
        # İstatistikleri güncelle
        self.total_reward += reward
        if action == true_label:
            self.correct_predictions += 1
            
        # Sonraki adıma geç
        self.current_step += 1
        
        # Episode bitmiş mi?
        done = self.current_step >= self.episode_length
        
        # Info dictionary
        info = {
            'true_label': true_label,
            'predicted_label': action,
            'accuracy': self.correct_predictions / max(self.current_step, 1),
            'total_reward': self.total_reward,
            'agent_type': self.agent_type.value
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, action, true_label, current_data):
        """Ödül fonksiyonu - ajan tipine göre farklılaştırılmış"""
        base_reward = 0
        
        # Temel doğruluk ödülü
        if action == true_label:
            base_reward = 1.0
        else:
            base_reward = -1.0
            
        # Ajan tipine göre ek ödüller/cezalar
        if self.agent_type == AgentType.SIMPLE:
            # Basit ajan: Sadece doğruluk
            return base_reward
            
        elif self.agent_type == AgentType.INTERMEDIATE:
            # Orta ajan: Güven seviyesine göre ek ödül
            confidence_bonus = 0
            if action == true_label:
                # Zor örneklerde doğru tahmin için bonus
                if current_data['bls_depth'] < current_data['bls_depth']:  # Düşük sinyal
                    confidence_bonus = 0.5
                    
            return base_reward + confidence_bonus
            
        else:  # COMPLEX
            # Karmaşık ajan: Detaylı ödül sistemi
            complexity_reward = 0
            
            if action == true_label:
                # Karmaşık durumlar için ek ödül
                flux_noise = current_data['pdcsap_flux_std'] / current_data['pdcsap_flux_mean']
                if flux_noise > 0.1:  # Yüksek gürültü
                    complexity_reward += 0.3
                    
                if current_data['bls_depth'] < 1000:  # Zayıf sinyal
                    complexity_reward += 0.4
                    
                # False positive'i önleme ödülü
                if true_label == 0 and action == 0:
                    complexity_reward += 0.2
                    
            else:
                # Yanlış tahmin cezası
                if true_label == 1 and action == 0:  # Missed detection
                    complexity_reward -= 0.5
                elif true_label == 0 and action == 1:  # False alarm
                    complexity_reward -= 0.3
                    
            return base_reward + complexity_reward
    
    def render(self, mode='human'):
        """Ortamı görselleştir"""
        if self.current_step == 0:
            return
            
        current_data = self.filtered_data.iloc[self.current_step - 1]
        
        print(f"Adım: {self.current_step}")
        print(f"Ajan Tipi: {self.agent_type.value}")
        print(f"Mevcut Doğruluk: {self.correct_predictions / self.current_step:.3f}")
        print(f"Toplam Ödül: {self.total_reward:.3f}")
        print(f"Mevcut Örnek ID: {current_data['id']}")
        print(f"Gerçek Label: {current_data['label']}")
        print("-" * 50)

class MultiAgentEnvironment:
    """Üç ajan tipini yöneten ana ortam"""
    
    def __init__(self, fuzzy_system, data_df):
        self.fuzzy_system = fuzzy_system
        self.data_df = data_df
        
        # Üç farklı ortam oluştur
        self.environments = {
            AgentType.SIMPLE: ExoplanetRLEnvironment(
                fuzzy_system, data_df, AgentType.SIMPLE
            ),
            AgentType.INTERMEDIATE: ExoplanetRLEnvironment(
                fuzzy_system, data_df, AgentType.INTERMEDIATE
            ),
            AgentType.COMPLEX: ExoplanetRLEnvironment(
                fuzzy_system, data_df, AgentType.COMPLEX
            )
        }
        
    def get_environment(self, agent_type):
        """Belirli ajan tipi için ortamı döndür"""
        return self.environments[agent_type]
    
    def get_all_environments(self):
        """Tüm ortamları döndür"""
        return self.environments
    
    def reset_all(self):
        """Tüm ortamları sıfırla"""
        observations = {}
        for agent_type, env in self.environments.items():
            observations[agent_type] = env.reset()
        return observations
    
    def get_environment_stats(self):
        """Ortam istatistiklerini döndür"""
        stats = {}
        for agent_type, env in self.environments.items():
            stats[agent_type.value] = {
                'data_size': len(env.filtered_data),
                'observation_space': env.observation_space.shape,
                'action_space': env.action_space.n,
                'positive_samples': sum(env.filtered_data['label']),
                'negative_samples': len(env.filtered_data) - sum(env.filtered_data['label'])
            }
        return stats

def test_environment():
    """Ortam testi"""
    from fuzzy_logic_system import ExoplanetFuzzySystem
    
    # Veriyi yükle
    df = pd.read_csv('feature_extracted__k2_dataset_clean.csv')
    
    # Bulanık mantık sistemini oluştur
    fuzzy_system = ExoplanetFuzzySystem()
    
    # Multi-agent ortamını oluştur
    multi_env = MultiAgentEnvironment(fuzzy_system, df)
    
    # İstatistikleri göster
    print("Ortam İstatistikleri:")
    stats = multi_env.get_environment_stats()
    for agent_type, stat in stats.items():
        print(f"\n{agent_type.upper()} Ajan:")
        for key, value in stat.items():
            print(f"  {key}: {value}")
    
    # Her ajan tipi için kısa test
    print("\n" + "="*60)
    print("ORTAM TESTLERİ")
    print("="*60)
    
    for agent_type in [AgentType.SIMPLE, AgentType.INTERMEDIATE, AgentType.COMPLEX]:
        print(f"\n{agent_type.value.upper()} Ajan Testi:")
        env = multi_env.get_environment(agent_type)
        
        obs = env.reset()
        print(f"İlk gözlem şekli: {obs.shape}")
        print(f"İlk gözlem: {obs}")
        
        # 5 adım test et
        for step in range(5):
            action = env.action_space.sample()  # Rastgele aksiyon
            obs, reward, done, info = env.step(action)
            
            print(f"Adım {step+1}: Aksiyon={action}, Ödül={reward:.3f}, "
                  f"Doğru={info['true_label']}, Doğruluk={info['accuracy']:.3f}")
            
            if done:
                break

if __name__ == "__main__":
    test_environment()