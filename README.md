# Grid-based Zero-Latency Demo (Refactored)

This project demonstrates a grid-world with RL (DQN) training and an LSTM predictor for user actions, with optional visualization.

## Project layout

- `src/`
  - `constants.py` — Global constants, device selection, and path management (creates `param/`).
  - `visualizer.py` — Pygame drawing with headless support.
  - `world.py` — GridWorld environment and `encode_state`.
  - `models.py` — `QNetwork` (DQN) and `LSTMPredictor` models.
  - `replay.py` — ReplayBuffer and Transition.
  - `training.py` — RL training and LSTM data/training pipeline.
  - `gameplay.py` — Interactive game loop driven by LSTM predictions.
- `train.py` — Main entry to train RL+LSTM and save artifacts to `param/` with metadata.
- `visualize_models.py` — Visualize trained RL/LSTM policies.
- `play_game.py` — Human playable loop with LSTM prediction overlay.
- `param/` — Trained parameter files and metadata (auto-created).
- `requirements.txt` — Python dependencies.

## Quickstart (uv)

We recommend using [uv](https://github.com/astral-sh/uv) for fast Python envs and installs.

1) Install uv (macOS Homebrew):

```bash
brew install uv
```

2) Create a virtual environment and install deps:

```bash
# Create venv in .venv
uv venv .venv

# Install requirements into that venv
uv pip install -r requirements.txt
```

3) Run scripts with uv (auto-activates the venv):

```bash
# Train (auto-headless if no display)
uv run train.py

# Visualize (auto-picks latest metadata in param/ if --meta omitted)
uv run visualize_models.py --mode both --headless

# Explicit metadata
uv run visualize_models.py --meta param/training_meta_<timestamp>.json --mode both

# Quick smoke test (1 episode, 5 steps)
uv run visualize_models.py --headless --episodes 1 --steps 5

# Play (arrow keys)
uv run play_game.py --meta param/training_meta_<timestamp>.json
```

Notes:
- Headless: add `--headless` to demo scripts, or run without DISPLAY set.
- macOS pygame tip: if window creation fails, try `--headless` or ensure a graphical session (no SSH-only).
- Torch install varies by CUDA availability. The `torch` entry in `requirements.txt` installs CPU build via PyPI by default. For GPU, install a matching wheel separately.

## Saved artifacts

- RL model: `param/rl_net_<timestamp>.pth`
- LSTM model: `param/lstm_net_<timestamp>.pth`
- Metadata: `param/training_meta_<timestamp>.json`
