"""
Geliştirilmiş PPO Ajanları - Fuzzy Logic Çıktılarına Özelleştirilmiş
Her kural çıktısı için optimize edilmiş PPO ajanları
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from enum import Enum
from rl_environment import AgentType

class FuzzyOutputType(Enum):
    """Fuzzy logic çıktı tipleri"""
    VERY_LOW = "çok_düşük"
    LOW = "düşük" 
    MEDIUM = "orta"
    HIGH = "yüksek"
    VERY_HIGH = "çok_yüksek"

class SpecializedPPONetwork(nn.Module):
    """Özelleştirilmiş PPO ağı - fuzzy çıktı tipine göre optimize edilmiş"""
    
    def __init__(self, state_dim, action_dim, fuzzy_output_type, agent_type=AgentType.INTERMEDIATE):
        super(SpecializedPPONetwork, self).__init__()
        
        self.fuzzy_output_type = fuzzy_output_type
        self.agent_type = agent_type
        
        # Fuzzy çıktı tipine göre ağ mimarisi ayarla
        hidden_dims = self._get_architecture_for_output_type(fuzzy_output_type)
        
        # Actor network (policy)
        actor_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (value function)
        critic_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        critic_layers.append(nn.Linear(prev_dim, 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Fuzzy çıktı tipine göre özel katmanlar
        self._add_specialized_layers(state_dim, fuzzy_output_type)
    
    def _get_architecture_for_output_type(self, fuzzy_output_type):
        """Fuzzy çıktı tipine göre ağ mimarisi belirle"""
        architectures = {
            FuzzyOutputType.VERY_LOW: [32, 16],      # Basit ağ - kesin red kararları
            FuzzyOutputType.LOW: [64, 32],           # Orta ağ - düşük olasılık
            FuzzyOutputType.MEDIUM: [128, 64, 32],   # Karmaşık ağ - belirsizlik yönetimi
            FuzzyOutputType.HIGH: [64, 32],          # Orta ağ - yüksek olasılık
            FuzzyOutputType.VERY_HIGH: [32, 16]      # Basit ağ - kesin kabul kararları
        }
        return architectures.get(fuzzy_output_type, [64, 32])
    
    def _add_specialized_layers(self, state_dim, fuzzy_output_type):
        """Özel katmanlar ekle"""
        if fuzzy_output_type == FuzzyOutputType.MEDIUM:
            # Belirsizlik için attention mechanism - embed_dim num_heads'e bölünebilir olmalı
            if state_dim % 2 == 0:
                self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=2)
            else:
                # Tek sayı ise 1 head kullan
                self.attention = nn.MultiheadAttention(embed_dim=state_dim, num_heads=1)
        elif fuzzy_output_type in [FuzzyOutputType.VERY_LOW, FuzzyOutputType.VERY_HIGH]:
            # Kesin kararlar için confidence layer
            self.confidence_layer = nn.Linear(state_dim, 1)
    
    def forward(self, state):
        """İleri geçiş"""
        # Özel işlemler
        if hasattr(self, 'attention') and self.fuzzy_output_type == FuzzyOutputType.MEDIUM:
            # Attention mechanism uygula
            state_expanded = state.unsqueeze(0)  # Sequence dimension ekle
            attended_state, _ = self.attention(state_expanded, state_expanded, state_expanded)
            state = attended_state.squeeze(0)
        
        # Actor ve critic çıktıları
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        # Confidence score (eğer varsa)
        confidence = None
        if hasattr(self, 'confidence_layer'):
            confidence = torch.sigmoid(self.confidence_layer(state))
        
        return action_probs, state_value, confidence

class SpecializedPPOAgent:
    """Özelleştirilmiş PPO Ajanı"""
    
    def __init__(self, state_dim, action_dim, fuzzy_output_type, agent_type=AgentType.INTERMEDIATE, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        
        self.fuzzy_output_type = fuzzy_output_type
        self.agent_type = agent_type
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Fuzzy çıktı tipine göre hiperparametreleri ayarla
        self.lr, self.eps_clip, self.k_epochs = self._adjust_hyperparameters(fuzzy_output_type, lr, eps_clip, k_epochs)
        
        # Ağları oluştur
        self.policy = SpecializedPPONetwork(state_dim, action_dim, fuzzy_output_type, agent_type)
        self.policy_old = SpecializedPPONetwork(state_dim, action_dim, fuzzy_output_type, agent_type)
        
        # Eski politikayı kopyala
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training stats
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'accuracies': [],
            'confidence_scores': []
        }
    
    def _adjust_hyperparameters(self, fuzzy_output_type, base_lr, base_eps_clip, base_k_epochs):
        """Fuzzy çıktı tipine göre hiperparametreleri ayarla"""
        adjustments = {
            FuzzyOutputType.VERY_LOW: {
                'lr_multiplier': 1.2,      # Hızlı öğrenme - kesin red
                'eps_clip': 0.15,          # Daha az exploration
                'k_epochs': 3              # Az güncelleme
            },
            FuzzyOutputType.LOW: {
                'lr_multiplier': 1.1,
                'eps_clip': 0.18,
                'k_epochs': 4
            },
            FuzzyOutputType.MEDIUM: {
                'lr_multiplier': 0.8,      # Yavaş öğrenme - belirsizlik
                'eps_clip': 0.25,          # Daha fazla exploration
                'k_epochs': 6              # Fazla güncelleme
            },
            FuzzyOutputType.HIGH: {
                'lr_multiplier': 1.1,
                'eps_clip': 0.18,
                'k_epochs': 4
            },
            FuzzyOutputType.VERY_HIGH: {
                'lr_multiplier': 1.2,      # Hızlı öğrenme - kesin kabul
                'eps_clip': 0.15,          # Daha az exploration
                'k_epochs': 3              # Az güncelleme
            }
        }
        
        adj = adjustments.get(fuzzy_output_type, {'lr_multiplier': 1.0, 'eps_clip': base_eps_clip, 'k_epochs': base_k_epochs})
        
        return (base_lr * adj['lr_multiplier'], 
                adj['eps_clip'], 
                adj['k_epochs'])
    
    def select_action(self, state):
        """Aksiyon seç"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _, confidence = self.policy(state_tensor)
            
            # Fuzzy çıktı tipine göre aksiyon seçim stratejisi
            if self.fuzzy_output_type == FuzzyOutputType.VERY_LOW:
                # Kesin red - her zaman 0 (exoplanet değil)
                action = 0
            elif self.fuzzy_output_type == FuzzyOutputType.VERY_HIGH:
                # Kesin kabul - her zaman 1 (exoplanet)
                action = 1
            else:
                # Probabilistic seçim
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
            
            # Memory'ye kaydet
            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.logprobs.append(dist.log_prob(torch.tensor(action)).item() if 'dist' in locals() else 0.0)
            
            # Confidence score kaydet
            if confidence is not None:
                self.training_stats['confidence_scores'].append(confidence.item())
        
        return action
    
    def update(self):
        """Policy güncelle"""
        # Rewards hesapla
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert to tensors
        old_states = torch.stack([torch.FloatTensor(s) for s in self.memory.states])
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long)
        old_logprobs = torch.tensor(self.memory.logprobs, dtype=torch.float32)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.k_epochs):
            # Current policy evaluation
            action_probs, state_values, _ = self.policy(old_states)
            state_values = state_values.squeeze()
            
            # Action probabilities
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            
            # Ratio
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # Surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(state_values, rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear_memory()
        
        # Update stats
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        
        return avg_policy_loss, avg_value_loss

class PPOMemory:
    """PPO Memory"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class EnhancedMultiAgentPPOTrainer:
    """Geliştirilmiş Multi-Agent PPO Trainer"""
    
    def __init__(self, multi_env, fuzzy_system, max_episodes=1000):
        self.multi_env = multi_env
        self.fuzzy_system = fuzzy_system
        self.max_episodes = max_episodes
        
        # Her fuzzy çıktı tipi için özelleştirilmiş ajanlar
        self.specialized_agents = {}
        
        for fuzzy_output in FuzzyOutputType:
            for agent_type, env in multi_env.get_all_environments().items():
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                
                agent_key = f"{fuzzy_output.value}_{agent_type.value}"
                
                self.specialized_agents[agent_key] = SpecializedPPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    fuzzy_output_type=fuzzy_output,
                    agent_type=agent_type
                )
    
    def get_fuzzy_output_for_state(self, state, env):
        """Durum için fuzzy çıktı tipini belirle"""
        # Fuzzy system ile tahmin yap
        try:
            # State'i fuzzy system formatına çevir
            features = self._state_to_fuzzy_features(state, env)
            
            # Fuzzy prediction
            prediction = self.fuzzy_system.predict_single(features)
            
            # Prediction'ı fuzzy output type'a çevir
            if prediction <= 20:
                return FuzzyOutputType.VERY_LOW
            elif prediction <= 40:
                return FuzzyOutputType.LOW
            elif prediction <= 60:
                return FuzzyOutputType.MEDIUM
            elif prediction <= 80:
                return FuzzyOutputType.HIGH
            else:
                return FuzzyOutputType.VERY_HIGH
                
        except:
            # Fallback - orta seviye
            return FuzzyOutputType.MEDIUM
    
    def _state_to_fuzzy_features(self, state, env):
        """State'i fuzzy features'a çevir"""
        # Bu fonksiyon environment'a göre özelleştirilebilir
        # Şimdilik basit bir mapping kullanıyoruz
        if len(state) >= 5:
            return {
                'snr_transit': state[0] * 100,
                'beta_factor': state[1] * 100,
                'depth_consistency': state[2] * 100,
                'duty_cycle': state[3] * 100,
                'odd_even_diff': state[4] * 100
            }
        else:
            # Padding ile 5 feature'a çıkar
            padded_state = list(state) + [0.5] * (5 - len(state))
            return {
                'snr_transit': padded_state[0] * 100,
                'beta_factor': padded_state[1] * 100,
                'depth_consistency': padded_state[2] * 100,
                'duty_cycle': padded_state[3] * 100,
                'odd_even_diff': padded_state[4] * 100
            }
    
    def train_specialized_agents(self, episodes_per_combination=50):
        """Özelleştirilmiş ajanları eğit"""
        print("Özelleştirilmiş PPO Ajanları Eğitiliyor...")
        print("=" * 60)
        
        results = {}
        
        for agent_key, agent in self.specialized_agents.items():
            # Agent key format: "fuzzy_output_agent_type" (örn: "çok_düşük_simple")
            parts = agent_key.split('_')
            if len(parts) >= 2:
                # Son part agent type, geri kalanlar fuzzy output
                agent_type_str = parts[-1]
                fuzzy_output_str = '_'.join(parts[:-1])
            else:
                continue  # Geçersiz key format
            
            try:
                agent_type = AgentType(agent_type_str)
            except ValueError:
                print(f"Geçersiz agent type: {agent_type_str}")
                continue
            
            print(f"\n{fuzzy_output_str.upper()} - {agent_type_str.upper()} Ajan Eğitimi...")
            
            env = self.multi_env.get_environment(agent_type)
            
            episode_rewards = []
            
            for episode in range(episodes_per_combination):
                state = env.reset()
                episode_reward = 0
                
                while True:
                    # Fuzzy çıktı tipini belirle
                    current_fuzzy_output = self.get_fuzzy_output_for_state(state, env)
                    
                    # Eğer bu ajan bu fuzzy çıktı için eğitiliyorsa aksiyon al
                    if current_fuzzy_output.value == fuzzy_output_str:
                        action = agent.select_action(state)
                    else:
                        # Random action (bu durumda eğitim yapma)
                        action = env.action_space.sample()
                    
                    next_state, reward, done, info = env.step(action)
                    
                    if current_fuzzy_output.value == fuzzy_output_str:
                        agent.memory.rewards.append(reward)
                        agent.memory.is_terminals.append(done)
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Update agent (eğer memory'de veri varsa)
                if len(agent.memory.rewards) > 0:
                    agent.update()
                
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"  Episode {episode}, Avg Reward: {avg_reward:.2f}")
            
            results[agent_key] = {
                'episode_rewards': episode_rewards,
                'final_avg_reward': np.mean(episode_rewards[-10:])
            }
        
        return results

def main():
    """Test fonksiyonu"""
    print("Geliştirilmiş PPO Ajanları Test Ediliyor...")
    
    # Test için basit bir örnek
    for fuzzy_output in FuzzyOutputType:
        print(f"\n{fuzzy_output.value} için ağ mimarisi test ediliyor...")
        
        network = SpecializedPPONetwork(
            state_dim=6, 
            action_dim=2, 
            fuzzy_output_type=fuzzy_output
        )
        
        # Test input
        test_input = torch.randn(1, 6)
        action_probs, state_value, confidence = network(test_input)
        
        print(f"  Action probs: {action_probs}")
        print(f"  State value: {state_value}")
        print(f"  Confidence: {confidence}")

if __name__ == "__main__":
    main()