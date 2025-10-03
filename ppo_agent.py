import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
from typing import Dict, List, Tuple
from rl_environment import AgentType, MultiAgentEnvironment

class PPONetwork(nn.Module):
    """PPO için Actor-Critic ağı"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, agent_type=AgentType.INTERMEDIATE):
        super(PPONetwork, self).__init__()
        
        self.agent_type = agent_type
        
        # Ajan tipine göre ağ karmaşıklığını ayarla
        if agent_type == AgentType.SIMPLE:
            hidden_layers = [hidden_dim]
        elif agent_type == AgentType.INTERMEDIATE:
            hidden_layers = [hidden_dim, hidden_dim // 2]
        else:  # COMPLEX
            hidden_layers = [hidden_dim * 2, hidden_dim, hidden_dim // 2]
        
        # Shared layers
        layers = []
        input_dim = state_dim
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Ağırlıkları başlat"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 0.01)
            module.bias.data.zero_()
    
    def forward(self, state):
        """İleri geçiş"""
        shared_features = self.shared_layers(state)
        
        # Policy distribution
        action_probs = self.actor(shared_features)
        
        # State value
        state_value = self.critic(shared_features)
        
        return action_probs, state_value
    
    def get_action(self, state):
        """Aksiyon seç"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), state_value

class PPOAgent:
    """PPO Ajanı"""
    
    def __init__(self, state_dim, action_dim, agent_type=AgentType.INTERMEDIATE, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        
        self.agent_type = agent_type
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Ajan tipine göre hiperparametreleri ayarla
        if agent_type == AgentType.SIMPLE:
            self.lr = lr * 1.5  # Basit ajan için daha hızlı öğrenme
            self.k_epochs = 3
        elif agent_type == AgentType.INTERMEDIATE:
            self.lr = lr
            self.k_epochs = k_epochs
        else:  # COMPLEX
            self.lr = lr * 0.7  # Karmaşık ajan için daha dikkatli öğrenme
            self.k_epochs = k_epochs + 2
        
        # Ağları oluştur
        self.policy = PPONetwork(state_dim, action_dim, agent_type=agent_type)
        self.policy_old = PPONetwork(state_dim, action_dim, agent_type=agent_type)
        
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
            'accuracies': []
        }
    
    def select_action(self, state):
        """Aksiyon seç"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, state_value = self.policy_old.get_action(state)
            
        self.memory.states.append(state.squeeze(0))
        self.memory.actions.append(action)
        self.memory.log_probs.append(log_prob)
        self.memory.state_values.append(state_value.squeeze(0))
        
        return action
    
    def update(self):
        """PPO güncelleme"""
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        
        for reward, is_terminal in zip(reversed(self.memory.rewards), 
                                     reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Convert lists to tensors
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.tensor(self.memory.actions).detach()
        old_log_probs = torch.stack(self.memory.log_probs).detach()
        old_state_values = torch.stack(self.memory.state_values).detach()
        
        # Calculate advantages
        advantages = rewards - old_state_values.squeeze()
        
        # PPO update
        policy_losses = []
        value_losses = []
        
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            
            # Calculate ratios
            new_log_probs = dist.log_prob(old_actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), rewards)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.memory.clear()
        
        # Store training stats
        self.training_stats['policy_losses'].append(np.mean(policy_losses))
        self.training_stats['value_losses'].append(np.mean(value_losses))
        
        return np.mean(policy_losses), np.mean(value_losses)

class PPOMemory:
    """PPO için hafıza"""
    
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPOTrainer:
    """PPO eğitim yöneticisi"""
    
    def __init__(self, multi_env, max_episodes=1000, update_timestep=2000):
        self.multi_env = multi_env
        self.max_episodes = max_episodes
        self.update_timestep = update_timestep
        
        # Her ajan tipi için PPO ajanı oluştur
        self.agents = {}
        for agent_type, env in multi_env.get_all_environments().items():
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            self.agents[agent_type] = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_type=agent_type
            )
        
        # Training stats
        self.training_history = {agent_type: [] for agent_type in AgentType}
    
    def train_single_agent(self, agent_type, episodes=100):
        """Tek ajan eğitimi"""
        agent = self.agents[agent_type]
        env = self.multi_env.get_environment(agent_type)
        
        timestep = 0
        episode_rewards = []
        
        print(f"\n{agent_type.value.upper()} Ajan Eğitimi Başlıyor...")
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            correct_predictions = 0
            
            while True:
                timestep += 1
                
                # Aksiyon seç
                action = agent.select_action(state)
                
                # Adım at
                next_state, reward, done, info = env.step(action)
                
                # Hafızaya kaydet
                agent.memory.rewards.append(reward)
                agent.memory.is_terminals.append(done)
                
                # İstatistikleri güncelle
                episode_reward += reward
                episode_length += 1
                if info['predicted_label'] == info['true_label']:
                    correct_predictions += 1
                
                state = next_state
                
                # PPO güncelleme
                if timestep % self.update_timestep == 0:
                    policy_loss, value_loss = agent.update()
                    
                if done:
                    break
            
            # Episode istatistikleri
            accuracy = correct_predictions / episode_length if episode_length > 0 else 0
            episode_rewards.append(episode_reward)
            
            agent.training_stats['episode_rewards'].append(episode_reward)
            agent.training_stats['episode_lengths'].append(episode_length)
            agent.training_stats['accuracies'].append(accuracy)
            
            # Progress raporu
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_accuracy = np.mean(agent.training_stats['accuracies'][-50:])
                
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Accuracy: {avg_accuracy:.3f}")
        
        return agent.training_stats
    
    def train_all_agents(self, episodes_per_agent=200):
        """Tüm ajanları eğit"""
        print("Tüm Ajanların Eğitimi Başlıyor...")
        print("=" * 60)
        
        results = {}
        
        for agent_type in [AgentType.SIMPLE, AgentType.INTERMEDIATE, AgentType.COMPLEX]:
            results[agent_type] = self.train_single_agent(agent_type, episodes_per_agent)
            
        return results
    
    def evaluate_agents(self, test_episodes=50):
        """Ajanları değerlendir"""
        print("\nAjan Değerlendirmesi...")
        print("=" * 40)
        
        results = {}
        
        for agent_type, agent in self.agents.items():
            env = self.multi_env.get_environment(agent_type)
            
            total_rewards = []
            total_accuracies = []
            
            for episode in range(test_episodes):
                state = env.reset()
                episode_reward = 0
                correct_predictions = 0
                episode_length = 0
                
                while True:
                    # Greedy action selection (no exploration)
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action_probs, _ = agent.policy(state_tensor)
                        action = torch.argmax(action_probs, dim=1).item()
                    
                    next_state, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    if info['predicted_label'] == info['true_label']:
                        correct_predictions += 1
                    
                    state = next_state
                    
                    if done:
                        break
                
                accuracy = correct_predictions / episode_length if episode_length > 0 else 0
                total_rewards.append(episode_reward)
                total_accuracies.append(accuracy)
            
            avg_reward = np.mean(total_rewards)
            avg_accuracy = np.mean(total_accuracies)
            std_reward = np.std(total_rewards)
            std_accuracy = np.std(total_accuracies)
            
            results[agent_type] = {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy
            }
            
            print(f"{agent_type.value.upper()} Ajan:")
            print(f"  Ortalama Ödül: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"  Ortalama Doğruluk: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
            print()
        
        return results
    
    def plot_training_results(self):
        """Eğitim sonuçlarını görselleştir"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        colors = {'simple': 'blue', 'intermediate': 'green', 'complex': 'red'}
        
        for i, agent_type in enumerate([AgentType.SIMPLE, AgentType.INTERMEDIATE, AgentType.COMPLEX]):
            agent = self.agents[agent_type]
            color = colors[agent_type.value]
            
            # Episode rewards
            axes[0, i].plot(agent.training_stats['episode_rewards'], color=color, alpha=0.7)
            axes[0, i].set_title(f'{agent_type.value.title()} - Episode Rewards')
            axes[0, i].set_xlabel('Episode')
            axes[0, i].set_ylabel('Reward')
            axes[0, i].grid(True)
            
            # Moving average
            if len(agent.training_stats['episode_rewards']) > 10:
                window = min(50, len(agent.training_stats['episode_rewards']) // 4)
                moving_avg = np.convolve(agent.training_stats['episode_rewards'], 
                                       np.ones(window)/window, mode='valid')
                axes[0, i].plot(range(window-1, len(agent.training_stats['episode_rewards'])), 
                              moving_avg, color='black', linewidth=2, label=f'MA({window})')
                axes[0, i].legend()
            
            # Accuracies
            axes[1, i].plot(agent.training_stats['accuracies'], color=color, alpha=0.7)
            axes[1, i].set_title(f'{agent_type.value.title()} - Accuracy')
            axes[1, i].set_xlabel('Episode')
            axes[1, i].set_ylabel('Accuracy')
            axes[1, i].grid(True)
            
            # Moving average for accuracy
            if len(agent.training_stats['accuracies']) > 10:
                window = min(50, len(agent.training_stats['accuracies']) // 4)
                moving_avg = np.convolve(agent.training_stats['accuracies'], 
                                       np.ones(window)/window, mode='valid')
                axes[1, i].plot(range(window-1, len(agent.training_stats['accuracies'])), 
                              moving_avg, color='black', linewidth=2, label=f'MA({window})')
                axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_agents(self, filepath='ppo_agents.pkl'):
        """Ajanları kaydet"""
        save_data = {}
        for agent_type, agent in self.agents.items():
            save_data[agent_type.value] = {
                'state_dict': agent.policy.state_dict(),
                'training_stats': agent.training_stats,
                'agent_type': agent_type
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Ajanlar {filepath} dosyasına kaydedildi.")
    
    def load_agents(self, filepath='ppo_agents.pkl'):
        """Ajanları yükle"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        for agent_type_str, data in save_data.items():
            agent_type = AgentType(agent_type_str)
            if agent_type in self.agents:
                self.agents[agent_type].policy.load_state_dict(data['state_dict'])
                self.agents[agent_type].policy_old.load_state_dict(data['state_dict'])
                self.agents[agent_type].training_stats = data['training_stats']
        
        print(f"Ajanlar {filepath} dosyasından yüklendi.")

def main():
    """Ana eğitim fonksiyonu"""
    from fuzzy_logic_system import ExoplanetFuzzySystem
    import pandas as pd
    
    # Veriyi yükle
    df = pd.read_csv('feature_extracted__k2_dataset_clean.csv')
    
    # Bulanık mantık sistemini oluştur
    fuzzy_system = ExoplanetFuzzySystem()
    
    # Multi-agent ortamını oluştur
    multi_env = MultiAgentEnvironment(fuzzy_system, df)
    
    # PPO trainer'ı oluştur
    trainer = PPOTrainer(multi_env, max_episodes=300)
    
    # Tüm ajanları eğit
    training_results = trainer.train_all_agents(episodes_per_agent=200)
    
    # Ajanları değerlendir
    evaluation_results = trainer.evaluate_agents(test_episodes=50)
    
    # Sonuçları görselleştir
    trainer.plot_training_results()
    
    # Ajanları kaydet
    trainer.save_agents('trained_ppo_agents.pkl')
    
    return trainer, training_results, evaluation_results

if __name__ == "__main__":
    trainer, training_results, evaluation_results = main()