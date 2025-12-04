import numpy as np
import torch
import pygame

from .constants import DEVICE, SEQ_LEN, NOOP, MOVE_COOLDOWN, ACTION_HISTORY, FPS, RTT_FRAMES
from .world import GridWorld, encode_state
from .visualizer import Visualizer


def run_game(viz: Visualizer, lstm_model):
    world = GridWorld().reset()
    state_hist = []
    past_actions = []
    delayed_u_actions = []
    server_send_buf = []

    CLIENT_ONE_WAY = RTT_FRAMES // 2
    for _ in range(SEQ_LEN):
        state_hist.append(encode_state(world, []))

    viz.speed_mode = 2
    running = True
    frame = 0
    total_preds, correct_preds = 0, 0
    input_cooldown_timer = 0

    while running:
        frame += 1
        viz.handle_speed_input()

        keys = pygame.key.get_pressed() if not viz.headless else []
        ua = None
        if input_cooldown_timer <= 0 and not viz.headless:
            if keys[pygame.K_UP]: ua = 0
            elif keys[pygame.K_DOWN]: ua = 1
            elif keys[pygame.K_LEFT]: ua = 2
            elif keys[pygame.K_RIGHT]: ua = 3
            if ua is not None: input_cooldown_timer = MOVE_COOLDOWN
        else:
            ua = None
            input_cooldown_timer -= 1

        curr_act = ua if ua is not None else NOOP

        past_actions.append(curr_act)
        if len(past_actions) > ACTION_HISTORY:
            past_actions.pop(0)
        delayed_u_actions.append([CLIENT_ONE_WAY, curr_act])

        # Client
        for buf in server_send_buf:
            buf[0] -= 1
        arrived = [b for b in server_send_buf if b[0] <= 0]
        server_send_buf = [b for b in server_send_buf if b[0] > 0]
        latest_pred = arrived[-1] if arrived else None

        match = False
        if latest_pred and curr_act != NOOP:
            total_preds += 1
            if curr_act in latest_pred[1]:
                correct_preds += 1
                match = True

        if curr_act != NOOP:
            world.client_apply(curr_act)

        # Server
        for item in delayed_u_actions:
            item[0] -= 1
        arrived_acts = [a for a in delayed_u_actions if a[0] <= 0]
        delayed_u_actions = [a for a in delayed_u_actions if a[0] > 0]
        if arrived_acts:
            s_act = arrived_acts[-1][1]
            if s_act != NOOP:
                world.step(s_act)

        # Predict
        curr_enc = encode_state(world, past_actions)
        state_hist.append(curr_enc)
        if len(state_hist) > 200:
            state_hist.pop(0)

        seq = torch.tensor([state_hist[-SEQ_LEN:]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(lstm_model(seq), dim=1)[0].cpu().numpy()
        top2 = probs.argsort()[-2:][::-1].tolist()
        server_send_buf.append([CLIENT_ONE_WAY, top2])

        if frame % (FPS * 5) == 0:
            world.cx, world.cy = world.ax, world.ay

        acc = (correct_preds / total_preds * 100) if total_preds > 0 else 0.0
        viz.draw(world, "3. Play", f"Acc: {acc:.1f}%", f"Match: {match}")
        if viz.speed_mode != 4 and not viz.headless:
            viz.wait_frame()
