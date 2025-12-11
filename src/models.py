import torch
import torch.nn as nn
from .constants import NUM_ACTIONS


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
