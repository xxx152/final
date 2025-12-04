"""Interactive play script using trained LSTM predictions.

Player controls the avatar with arrow keys. The LSTM model (loaded from
file) predicts next action probabilities based on recent encoded states.
Top-2 predicted actions are shown; accuracy tracking is displayed when
the player's action matches one of the top predictions.

Usage:
  python play_game.py --lstm lstm_net_1700000000.pth --meta training_meta_1700000000.json
  python play_game.py --meta training_meta_1700000000.json

If --meta is supplied and --lstm omitted, it auto-loads the lstm path in metadata.
"""

import argparse
import json
import os
import numpy as np
import torch
import pygame

from src.visualizer import Visualizer
from src.world import GridWorld, encode_state
from src.models import LSTMPredictor
from src.constants import (
    INPUT_DIM,
    SEQ_LEN,
    DEVICE,
    NOOP,
    MOVE_COOLDOWN,
    ACTION_HISTORY,
    PARAM_DIR,
    FPS,
    RTT_FRAMES,
)


def load_lstm(path: str):
    model = LSTMPredictor(INPUT_DIM).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run(viz: Visualizer, lstm_model: LSTMPredictor):
    world = GridWorld().reset()
    state_hist = [encode_state(world, []) for _ in range(SEQ_LEN)]
    past_actions = []

    # Use RTT_FRAMES from constants for network delay simulation
    import time
    CLIENT_DELAY = RTT_FRAMES // 2  # Half of RTT is one-way delay
    
    delayed_user_actions = []
    server_pred_buffer = []

    frame = 0
    total_preds = 0
    correct_preds = 0
    input_cooldown = 0

    viz.speed_mode = 2 if not viz.headless else 0  # Start from SLOW

    # FPS tracking
    fps_timer = time.perf_counter()
    fps_counter = 0
    current_fps = 0.0

    running = True
    while running:
        frame += 1
        fps_counter += 1
        
        # Update FPS every second
        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            current_fps = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer = now
        
        viz.handle_speed_input()

        if not viz.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Predict next action probabilities from recent states
        seq = torch.tensor([state_hist[-SEQ_LEN:]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(lstm_model(seq), dim=1)[0].cpu().numpy()
        top2 = probs.argsort()[-2:][::-1].tolist()
        predicted_action = int(np.argmax(probs))
        # Send prediction to client buffer with delay (for accuracy stats)
        server_pred_buffer.append([CLIENT_DELAY, top2])

        # Keyboard input
        # Keyboard input with continuous movement while a key is held
        ua = None
        if not viz.headless:
            keys = pygame.key.get_pressed()
            desired_dir = None
            if keys[pygame.K_UP]:
                desired_dir = 0
            elif keys[pygame.K_DOWN]:
                desired_dir = 1
            elif keys[pygame.K_LEFT]:
                desired_dir = 2
            elif keys[pygame.K_RIGHT]:
                desired_dir = 3

            if input_cooldown <= 0:
                # If a direction key is currently held, move now
                if desired_dir is not None:
                    ua = desired_dir
                    input_cooldown = MOVE_COOLDOWN
            else:
                # Count down the cooldown; when it reaches 0 and the key is still held, repeat the move
                input_cooldown -= 1
                if input_cooldown <= 0 and desired_dir is not None:
                    ua = desired_dir
                    input_cooldown = MOVE_COOLDOWN

        # AI assist: if enabled and user idle, apply predicted action
        if ua is None and input_cooldown <= 0 and viz.ai_assist_enabled:
            ua = predicted_action
            input_cooldown = MOVE_COOLDOWN

        curr_act = ua if ua is not None else NOOP
        past_actions.append(curr_act)
        if len(past_actions) > ACTION_HISTORY:
            past_actions.pop(0)
        delayed_user_actions.append([CLIENT_DELAY, curr_act])

        # Apply latest arrived server prediction (for accuracy stats only)
        for item in server_pred_buffer:
            item[0] -= 1
        arrived_preds = [p for p in server_pred_buffer if p[0] <= 0]
        server_pred_buffer = [p for p in server_pred_buffer if p[0] > 0]
        latest_pred = arrived_preds[-1] if arrived_preds else None

        match = False
        if latest_pred and curr_act != NOOP:
            total_preds += 1
            if curr_act in latest_pred[1]:
                correct_preds += 1
                match = True

        # Client applies immediate movement
        if curr_act != NOOP:
            world.client_apply(curr_act)

        # Server applies delayed user actions
        for item in delayed_user_actions:
            item[0] -= 1
        arrived_actions = [a for a in delayed_user_actions if a[0] <= 0]
        delayed_user_actions = [a for a in delayed_user_actions if a[0] > 0]
        if arrived_actions:
            act = arrived_actions[-1][1]
            if act != NOOP:
                world.step(act)

        # Encode latest state
        curr_enc = encode_state(world, past_actions)
        state_hist.append(curr_enc)
        if len(state_hist) > 400:
            state_hist.pop(0)

        # Periodically re-sync client avatar to authoritative server pos
        if frame % 300 == 0:
            world.cx, world.cy = world.ax, world.ay

        acc = (correct_preds / total_preds * 100) if total_preds else 0.0
        info = f"Acc={acc:.1f}% FPS={current_fps:.1f}"
        extra = f"Top2={top2}\nMatch={match}\nScore={world.score}\nLegend: Green=Server (authoritative), Blue=Client (predicted/applied)"
        # Show zero latency when prediction matches (client-side prediction success)
        effective_delay = 0 if match else CLIENT_DELAY
        viz.draw(world, "Play (Human + LSTM Predict)", info, extra, client_delay=effective_delay)
        if viz.speed_mode != 4 and not viz.headless:
            viz.wait_frame()
    print("Session ended. Final accuracy: %.2f%%" % acc)


def main():
    parser = argparse.ArgumentParser(description="Play interactively with LSTM predictions.")
    parser.add_argument("--meta", type=str, default=None, help="Metadata JSON (auto-lstm path). If omitted, latest meta in param/ will be used if present.")
    parser.add_argument("--lstm", type=str, default=None, help="Path to lstm model .pth")
    parser.add_argument("--headless", action="store_true", help="Force headless (no window).")
    parser.add_argument("--hide-client-dot", action="store_true", help="Hide the blue client dot in visualization.")
    parser.add_argument("--fast", action="store_true", help="AI visualization fast mode (no frame wait).")
    args = parser.parse_args()

    lstm_path = args.lstm
    # If meta provided, use it; else try latest in PARAM_DIR
    meta_path = None
    if args.meta and os.path.isfile(args.meta):
        meta_path = args.meta
    else:
        try:
            metas = [f for f in os.listdir(PARAM_DIR) if f.startswith("training_meta_") and f.endswith(".json")]
            if metas:
                metas.sort(reverse=True)
                meta_path = os.path.join(PARAM_DIR, metas[0])
        except FileNotFoundError:
            meta_path = None

    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not lstm_path:
            lstm_path = os.path.join(PARAM_DIR, meta.get("artifacts", {}).get("lstm", ""))

    if not lstm_path:
        raise SystemExit("Must provide --lstm or --meta with lstm path.")

    if not os.path.isabs(lstm_path):
        lstm_path = os.path.join(PARAM_DIR, lstm_path)

    lstm_model = load_lstm(lstm_path)
    viz = Visualizer(headless=args.headless)
    # Apply visualization toggles
    if args.hide_client_dot:
        viz.show_client_dot = False
    if args.fast and not viz.headless:
        viz.speed_mode = 4
    print(f"Loaded LSTM from {lstm_path}. Headless={viz.headless}")
    run(viz, lstm_model)


if __name__ == "__main__":
    main()
