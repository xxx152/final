"""Visualization script for trained RL and LSTM models.

Usage examples:
  python visualize_models.py --meta training_meta_1700000000.json --mode rl
  python visualize_models.py --rl rl_net_1700000000.pth --lstm lstm_net_1700000000.pth --mode both

Modes:
  rl    : Use the trained RL Q-network greedily to act, visualize performance.
  lstm  : Use the trained LSTM predictor to act (argmax policy) and visualize.
  both  : Run RL first for some steps then LSTM policy for same steps.

Requires display unless --headless specified or DISPLAY missing.
"""

import argparse
import os
import time
import json
import numpy as np
import torch
from typing import Optional

# Import from src package
from src.visualizer import Visualizer
from src.world import GridWorld, encode_state
from src.models import QNetwork, LSTMPredictor
from src.constants import INPUT_DIM, NUM_ACTIONS, SEQ_LEN, DEVICE, PARAM_DIR


def load_artifacts(rl_path: Optional[str], lstm_path: Optional[str]):
    rl_net = None
    lstm_net = None
    if rl_path:
        rl_net = QNetwork(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
        rl_net.load_state_dict(torch.load(rl_path, map_location=DEVICE))
        rl_net.eval()
    if lstm_path:
        lstm_net = LSTMPredictor(INPUT_DIM).to(DEVICE)
        lstm_net.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        lstm_net.eval()
    return rl_net, lstm_net


def demo_rl(viz: Visualizer, rl_net: QNetwork, episodes: int, steps: int):
    world = GridWorld()
    for ep in range(episodes):
        world.reset()
        past_actions = []
        state = encode_state(world, past_actions)
        score = 0
        for step in range(steps):
            with torch.no_grad():
                action = rl_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()
            reward, _ = world.step(action)
            score += reward if reward > 0 else 0
            past_actions.append(action)
            if len(past_actions) > 6:
                past_actions.pop(0)
            state = encode_state(world, past_actions)
            viz.handle_speed_input()
            viz.draw(world, f"RL Demo Ep {ep+1}/{episodes}", f"Step {step+1}/{steps}", f"Score={world.score}")
            if viz.speed_mode != 4:
                viz.wait_frame()


def demo_lstm(viz: Visualizer, lstm_net: LSTMPredictor, episodes: int, steps: int):
    world = GridWorld()
    for ep in range(episodes):
        world.reset()
        past_actions = []
        state_hist = [encode_state(world, past_actions) for _ in range(SEQ_LEN)]
        for step in range(steps):
            seq = torch.tensor([state_hist[-SEQ_LEN:]], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(lstm_net(seq), dim=1)[0].cpu().numpy()
            action = int(np.argmax(probs))
            reward, _ = world.step(action)
            past_actions.append(action)
            if len(past_actions) > 6:
                past_actions.pop(0)
            next_state = encode_state(world, past_actions)
            state_hist.append(next_state)
            viz.handle_speed_input()
            info = f"Step {step+1}/{steps} | PredAct={action} | Reward={reward:.1f}"
            extra = f"Score={world.score} | TopProb={probs[action]:.2f}"
            viz.draw(world, f"LSTM Demo Ep {ep+1}/{episodes}", info, extra)
            if viz.speed_mode != 4:
                viz.wait_frame()


def main():
    parser = argparse.ArgumentParser(description="Visualize trained RL/LSTM agents.")
    parser.add_argument("--meta", type=str, default=None, help="Path to metadata JSON (auto-populates rl/lstm paths). If omitted, latest meta in param/ will be used if present.")
    parser.add_argument("--rl", type=str, default=None, help="Path to RL .pth file.")
    parser.add_argument("--lstm", type=str, default=None, help="Path to LSTM .pth file.")
    parser.add_argument("--mode", choices=["rl", "lstm", "both"], default="rl", help="Demo mode.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per mode.")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode.")
    parser.add_argument("--headless", action="store_true", help="Force headless (no drawing).")
    args = parser.parse_args()

    rl_path = args.rl
    lstm_path = args.lstm

    # If meta provided, use it; else try latest in PARAM_DIR
    meta_path = None
    if args.meta and os.path.isfile(args.meta):
        meta_path = args.meta
    else:
        # auto-pick latest metadata in param/
        try:
            metas = [f for f in os.listdir(PARAM_DIR) if f.startswith("training_meta_") and f.endswith(".json")]
            if metas:
                metas.sort(reverse=True)
                meta_path = os.path.join(PARAM_DIR, metas[0])
        except FileNotFoundError:
            meta_path = None

    if meta_path:
        # Load metadata from resolved meta_path (not args.meta which may be None)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        artifacts = meta.get("artifacts", {})
        # Only populate paths if keys exist; avoid joining empty strings
        if rl_path is None and artifacts.get("rl"):
            rl_path = os.path.join(PARAM_DIR, artifacts["rl"])
        if lstm_path is None and artifacts.get("lstm"):
            lstm_path = os.path.join(PARAM_DIR, artifacts["lstm"])

    # Default to PARAM_DIR if only basenames provided
    if rl_path and not os.path.isabs(rl_path):
        rl_path = os.path.join(PARAM_DIR, rl_path)
    if lstm_path and not os.path.isabs(lstm_path):
        lstm_path = os.path.join(PARAM_DIR, lstm_path)

    if args.mode in ("rl", "both") and not rl_path:
        parser.error("RL mode requested but --rl path not provided (or not found via --meta).")
    if args.mode in ("lstm", "both") and not lstm_path:
        parser.error("LSTM mode requested but --lstm path not provided (or not found via --meta).")

    rl_net, lstm_net = load_artifacts(rl_path, lstm_path)
    viz = Visualizer(headless=args.headless)

    print(f"Mode={args.mode} | Episodes={args.episodes} | Steps={args.steps} | Headless={viz.headless}")

    if args.mode == "rl":
        demo_rl(viz, rl_net, args.episodes, args.steps)
    elif args.mode == "lstm":
        demo_lstm(viz, lstm_net, args.episodes, args.steps)
    else:  # both
        print("--- RL Demo ---")
        demo_rl(viz, rl_net, args.episodes, args.steps)
        print("--- LSTM Demo ---")
        demo_lstm(viz, lstm_net, args.episodes, args.steps)

    print("Demo finished.")


if __name__ == "__main__":
    main()
