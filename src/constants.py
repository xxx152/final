import os
import torch

# Device selection
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
)

# Grid/Screen
GRID_W, GRID_H = 16, 12
CELL = 48
SCREEN_W_BASE = GRID_W * CELL
SCREEN_W = SCREEN_W_BASE + 300
SCREEN_H = GRID_H * CELL

# Timing/Gameplay
FPS = 60
RTT_FRAMES = 15
MOVE_COOLDOWN = 5
NUM_REWARDS = 5

# Actions
NUM_ACTIONS = 5
NOOP = 4
ACTION_SPACE = NUM_ACTIONS
ACTION_HISTORY = 6
SEQ_LEN = 8

# State encoding
REWARD_K = 5
OBS_K = 20
BASE_FEATURE_DIM = 2 + 2 * REWARD_K + 2 * OBS_K
INPUT_DIM = BASE_FEATURE_DIM + (ACTION_HISTORY * ACTION_SPACE)

# Training params
RL_EPISODES = 3000     
RL_STEPS_PER_EP = 720       
DATA_COLLECTION_SIZE = 150000
LSTM_TRAIN_EPOCHS = 40
MAP_CHANGE_FREQ = 150
BATCH_SIZE = 256
REPLAY_CAPACITY = 200000
GAMMA = 0.99
LR = 0.0002
PROGRESS_SCALE = 1.0
TARGET_UPDATE_FREQ = 800
# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PARAM_DIR = os.path.join(ROOT_DIR, "param")  # trained parameters live here
os.makedirs(PARAM_DIR, exist_ok=True)
