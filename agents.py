# agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class A2CAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.value = ValueNetwork(input_dim)
        self.optim = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        action = torch.distributions.Categorical(probs).sample().item()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0) if next_state is not None else torch.zeros_like(state)
        reward = torch.tensor([reward], dtype=torch.float32)

        log_prob = torch.log(self.policy(state)[0][action])
        value = self.value(state)
        next_value = self.value(next_state)
        target = reward + self.gamma * next_value * (1 - int(done))
        advantage = target - value

        loss = -log_prob * advantage.detach() + advantage.pow(2)

        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()


class PPOAgent:
    def __init__(self, input_dim, action_dim, gamma=0.99, lr=1e-3, clip=0.2):
        self.gamma = gamma
        self.clip = clip
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.value = ValueNetwork(input_dim)
        self.optim = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, trajectories, epochs=4):
        states, actions, rewards, dones, old_log_probs = trajectories
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.stack(old_log_probs)

        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        for _ in range(epochs):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs.detach())
            adv = returns - self.value(states).squeeze()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = adv.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
