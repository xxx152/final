"""Train RL and LSTM, save parameters under param/ and write metadata JSON.

Usage:
    python train.py [--headless] [--episodes N] [--steps M] [--epochs E] [--batch B] [--lr LR] [--gamma G] [--config PATH]
"""

import os
import time
import json
import argparse
import torch

import src.constants as C
from src.visualizer import Visualizer
from src.training import train_rl_agent, train_lstm_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Train RL + LSTM and save artifacts to param/.")
    p.add_argument("--headless", action="store_true", help="Force headless training (no windows).")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config to override training and env params.")
    p.add_argument("--episodes", type=int, default=C.RL_EPISODES, help="RL training episodes.")
    p.add_argument("--steps", type=int, default=C.RL_STEPS_PER_EP, help="Steps per RL episode.")
    p.add_argument("--epochs", type=int, default=C.LSTM_TRAIN_EPOCHS, help="LSTM training epochs.")
    p.add_argument("--batch", type=int, default=C.BATCH_SIZE, help="Mini-batch size for RL updates and LSTM.")
    p.add_argument("--lr", type=float, default=C.LR, help="Learning rate for RL (QNet) and reported in metadata.")
    p.add_argument("--gamma", type=float, default=C.GAMMA, help="Discount factor.")
    p.add_argument("--replay", type=int, default=C.REPLAY_CAPACITY, help="Replay buffer capacity.")
    p.add_argument("--data-size", type=int, default=C.DATA_COLLECTION_SIZE, help="Expert data size for LSTM.")
    p.add_argument("--map-change", type=int, default=C.MAP_CHANGE_FREQ, help="Map change frequency (episodes).")
    return p.parse_args()


def apply_overrides(args):
    # Override constants at runtime so training uses CLI values
    C.RL_EPISODES = args.episodes
    C.RL_STEPS_PER_EP = args.steps
    C.LSTM_TRAIN_EPOCHS = args.epochs
    C.BATCH_SIZE = args.batch
    C.LR = args.lr
    C.GAMMA = args.gamma
    C.REPLAY_CAPACITY = args.replay
    C.DATA_COLLECTION_SIZE = args.data_size
    C.MAP_CHANGE_FREQ = args.map_change


def apply_json_config(path: str):
        """Load a JSON config and apply to constants. Unknown keys are ignored.
        Example keys:
            {
                "training": {"episodes": 200, "steps": 300, "epochs": 20, "batch": 128, "lr": 0.0002, "gamma": 0.95, "replay": 100000, "data_size": 50000, "map_change": 120},
                "env": {"grid_w": 16, "grid_h": 12, "cell": 48, "num_rewards": 5, "rtt_frames": 15, "fps": 60}
            }
        """
        try:
                with open(path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
        except Exception as e:
                print(f"Failed to read config '{path}': {e}")
                return

        tr = cfg.get("training", {})
        C.RL_EPISODES = tr.get("episodes", C.RL_EPISODES)
        C.RL_STEPS_PER_EP = tr.get("steps", C.RL_STEPS_PER_EP)
        C.LSTM_TRAIN_EPOCHS = tr.get("epochs", C.LSTM_TRAIN_EPOCHS)
        C.BATCH_SIZE = tr.get("batch", C.BATCH_SIZE)
        C.LR = tr.get("lr", C.LR)
        C.GAMMA = tr.get("gamma", C.GAMMA)
        C.REPLAY_CAPACITY = tr.get("replay", C.REPLAY_CAPACITY)
        C.DATA_COLLECTION_SIZE = tr.get("data_size", C.DATA_COLLECTION_SIZE)
        C.MAP_CHANGE_FREQ = tr.get("map_change", C.MAP_CHANGE_FREQ)

        env = cfg.get("env", {})
        C.GRID_W = env.get("grid_w", C.GRID_W)
        C.GRID_H = env.get("grid_h", C.GRID_H)
        C.CELL = env.get("cell", C.CELL)
        C.NUM_REWARDS = env.get("num_rewards", C.NUM_REWARDS)
        C.RTT_FRAMES = env.get("rtt_frames", C.RTT_FRAMES)
        C.FPS = env.get("fps", C.FPS)


def main():
    args = parse_args()
    # First apply JSON config if provided, then CLI overrides take precedence
    if args.config:
        apply_json_config(args.config)
    apply_overrides(args)

    # auto headless if no DISPLAY and not overridden
    headless = args.headless or (not bool(os.environ.get("DISPLAY")))
    viz = Visualizer(headless=headless)
    print(f"Using device: {C.DEVICE}")
    print("Start Phase 1... (Training RL Agent)")
    viz.speed_mode = 3
    print(f'{args}')
    rl_net = train_rl_agent(viz)

    print("Start Phase 2... (Training LSTM)")
    viz.speed_mode = 3
    lstm_net = train_lstm_pipeline(viz, rl_net)

    # Save artifacts
    timestamp = int(time.time())
    rl_path = os.path.join(C.PARAM_DIR, f"rl_net_{timestamp}.pth")
    lstm_path = os.path.join(C.PARAM_DIR, f"lstm_net_{timestamp}.pth")
    meta_path = os.path.join(C.PARAM_DIR, f"training_meta_{timestamp}.json")

    torch.save(rl_net.state_dict(), rl_path)
    torch.save(lstm_net.state_dict(), lstm_path)
    meta = {
        "timestamp": timestamp,
        "device": str(C.DEVICE),
        "grid": {"w": C.GRID_W, "h": C.GRID_H},
        "cell": C.CELL,
        "num_rewards": C.NUM_REWARDS,
        "rtt_frames": C.RTT_FRAMES,
        "fps": C.FPS,
        "model": {
            "input_dim": C.INPUT_DIM,
            "action_space": C.ACTION_SPACE,
            "action_history": C.ACTION_HISTORY,
            "seq_len": C.SEQ_LEN,
        },
        "training": {
            "rl_episodes": C.RL_EPISODES,
            "rl_steps_per_episode": C.RL_STEPS_PER_EP,
            "lstm_epochs": C.LSTM_TRAIN_EPOCHS,
            "replay_capacity": C.REPLAY_CAPACITY,
            "batch_size": C.BATCH_SIZE,
            "gamma": C.GAMMA,
            "lr": C.LR,
        },
        "artifacts": {"rl": os.path.basename(rl_path), "lstm": os.path.basename(lstm_path)},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved models to {rl_path}, {lstm_path} and metadata to {meta_path}.")


if __name__ == "__main__":
    main()
