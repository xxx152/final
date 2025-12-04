import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .constants import (
    INPUT_DIM,
    NUM_ACTIONS,
    DEVICE,
    RL_EPISODES,
    RL_STEPS_PER_EP,
    BATCH_SIZE,
    GAMMA,
    MAP_CHANGE_FREQ,
    ACTION_HISTORY,
    DATA_COLLECTION_SIZE,
    LSTM_TRAIN_EPOCHS,
)
from .world import GridWorld, encode_state
from .models import QNetwork, LSTMPredictor
from .replay import ReplayBuffer


def train_rl_agent(viz):
    world = GridWorld()
    q_net = QNetwork(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
    target_net = QNetwork(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(150000)

    score_history = []
    steps_done = 0
    # Early stopping: stop if no improvement for this many consecutive episodes
    best_score = -float('inf')
    no_improve_epochs = 0
    EARLY_STOP_PATIENCE = 4

    # Warm-up
    print("Warm-up replay buffer...")
    world.reset()
    warm_state = encode_state(world, [])
    for _ in range(15000):
        action = random.randint(0, NUM_ACTIONS - 1)
        reward, _ = world.step(action)
        next_state = encode_state(world, [action])
        replay_buffer.push(warm_state, action, reward, next_state, False)
        warm_state = next_state
        if random.random() < 0.05:
            world.reset()
            warm_state = encode_state(world, [])

    for ep in range(RL_EPISODES):
        if ep % MAP_CHANGE_FREQ == 0 and ep > 0:
            world.randomize_obstacles()
        world.reset()
        past_actions = []
        state = encode_state(world, past_actions)

        for step in range(RL_STEPS_PER_EP):
            steps_done += 1
            epsilon = 0.05 + 0.95 * math.exp(-steps_done / 800000)
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()

            base_reward, _ = world.step(action)
            reward = base_reward

            past_actions.append(action)
            if len(past_actions) > ACTION_HISTORY:
                past_actions.pop(0)
            next_state = encode_state(world, past_actions)

            replay_buffer.push(state, action, reward, next_state, step == RL_STEPS_PER_EP-1)
            state = next_state

            if len(replay_buffer) >= BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                batch_state = torch.FloatTensor(np.array(transitions.state)).to(DEVICE)
                batch_action = torch.LongTensor(transitions.action).unsqueeze(1).to(DEVICE)
                batch_reward = torch.FloatTensor(transitions.reward).unsqueeze(1).to(DEVICE)
                batch_next = torch.FloatTensor(np.array(transitions.next_state)).to(DEVICE)
                batch_done = torch.FloatTensor(transitions.done).unsqueeze(1).to(DEVICE)

                current_q = q_net(batch_state).gather(1, batch_action)
                with torch.no_grad():
                    next_actions = q_net(batch_next).argmax(1).unsqueeze(1)
                    next_q = target_net(batch_next).gather(1, next_actions)
                    target_q = batch_reward + GAMMA * next_q * (1.0 - batch_done)
                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

            if steps_done % 1000 == 0:
                target_net.load_state_dict(q_net.state_dict())

            if viz and getattr(viz, "speed_mode", 0) != 0:
                viz.draw_rl_status(world, ep, score_history, RL_EPISODES, epsilon, MAP_CHANGE_FREQ, reward)
                if viz.speed_mode != 4:
                    viz.wait_frame()

        score_history.append(world.score)

        if (ep + 1) % 50 == 0:
                    # Early-stop check: if no improvement for EARLY_STOP_PATIENCE consecutive episodes, break
            if np.mean(score_history[-50:]) > best_score:
                best_score = np.mean(score_history[-50:])
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"No improvement for {no_improve_epochs} episodes, stopping early at episode {ep+1}.")
                break
            print(f"Ep {ep+1} Score: {world.score}  Avg50: {np.mean(score_history[-50:]):.1f}")
        

    return q_net


def train_lstm_pipeline(viz, rl_agent):
    world = GridWorld()
    inputs, targets = [], []
    world.reset()
    past_actions = []
    state_hist = [encode_state(world, past_actions) for _ in range(8)]

    steps = 0
    map_timer = 0
    while steps < DATA_COLLECTION_SIZE:
        if viz:
            viz.handle_speed_input()
            if viz.speed_mode != 0:
                viz.draw(world, "2a. Gen Data (Fixed)", f"Steps: {steps}/{DATA_COLLECTION_SIZE}", "Collecting expert data...")
                if viz.speed_mode != 4:
                    viz.wait_frame()

        curr_vec = state_hist[-1]
        with torch.no_grad():
            q_vals = rl_agent(torch.FloatTensor(curr_vec).unsqueeze(0).to(DEVICE))
            expert_action = torch.argmax(q_vals).item()
        action = expert_action if random.random() < 0.9 else random.randint(0, NUM_ACTIONS - 1)
        world.step(action)
        past_actions.append(action)
        if len(past_actions) > ACTION_HISTORY:
            past_actions.pop(0)
        next_state = encode_state(world, past_actions)
        state_hist.append(next_state)
        state_hist.pop(0)

        inputs.append(np.array(state_hist[-8:]))
        targets.append(expert_action)

        steps += 1
        map_timer += 1
        if map_timer > 600:
            world.randomize_obstacles()
            world.reset()
            past_actions = []
            state_hist = [encode_state(world, past_actions) for _ in range(8)]
            map_timer = 0

    lstm = LSTMPredictor(INPUT_DIM).to(DEVICE)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss().to(DEVICE)
    ds = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(inputs)), torch.LongTensor(targets))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    loss_history = []
    for epoch in range(LSTM_TRAIN_EPOCHS):
        if viz:
            viz.handle_speed_input()
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(lstm(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        if viz and viz.speed_mode != 0:
            viz.draw_loss_graph(epoch, avg_loss, loss_history, LSTM_TRAIN_EPOCHS)
            if viz.speed_mode != 4:
                viz.wait_frame()

    return lstm
