# envs.py
import numpy as np
import pandas as pd

class ExoBanditEnv:
    """
    Contextual Bandit ortamı:
    - Gözlem: Flux değerleri (ve öznitelik transformasyonu)
    - Aksiyon: {0: yok, 1: var}
    - Ödül: doğruysa +1, yanlışsa -1
    """
    def __init__(self, dataframe, label_col="LABEL"):
        self.df = dataframe.copy()
        self.features = self.df.drop(columns=[label_col]).values
        self.labels = self.df[label_col].values
        self.n = len(self.df)
        self.idx = 0
        self.action_space = 2
        self.observation_space = self.features.shape[1]

    def reset(self):
        self.idx = np.random.randint(0, self.n)
        return self.features[self.idx]

    def step(self, action):
        # Label mapping (örnek: 2=gezegen var, 1=gezegen yok)
        label = 1 if self.labels[self.idx] == 2 else 0

        # Reward shaping: No Planet (0) doğru tahmin edilirse daha fazla ödül
        if action == label:
            reward = 2 if label == 0 else 1
        else:
            reward = -1

        done = True
        info = {"true_label": label}
        return self.features[self.idx], reward, done, info


class ExoSequenceEnv:
    """
    Sekans tabanlı ortam:
    - Gözlem: Flux serisi (sliding window)
    - Aksiyon: {0: yok, 1: var}
    """
    def __init__(self, dataframe, label_col="LABEL", window=5):
        self.df = dataframe.copy()
        self.features = self.df.drop(columns=[label_col]).values
        self.labels = self.df[label_col].values
        self.n = len(self.df)
        self.window = window
        self.idx = 0
        self.action_space = 2
        self.observation_space = window * self.features.shape[1]

    def reset(self):
        self.idx = 0
        return self.features[self.idx:self.idx+self.window].flatten()

    def step(self, action):
        true_label = int(self.labels[self.idx] == 1)
        reward = 1 if action == true_label else -1
        self.idx += 1
        done = self.idx >= (self.n - self.window)
        obs = None if done else self.features[self.idx:self.idx+self.window].flatten()
        info = {"true_label": true_label}
        return obs, reward, done, info
