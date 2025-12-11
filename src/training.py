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

    optimizer = optim.Adam(q_net.parameters(), lr=0.0002)
    replay_buffer = ReplayBuffer(200000)
    steps_history = []  # 紀錄每代的步數
    loss_history = []   # 紀錄訓練loss

    steps_done = 0
    max_steps_per_ep = 720  # 每代最多步數，避免無限循環

    warmup_episodes = max(1, RL_EPISODES // 3)  # 前1/3 逐步加障礙
    for ep in range(RL_EPISODES):
        # 障礙密度從 0 線性提升到原本的 0.05
        target_density = 0.04
        ramp = min(1.0, ep / warmup_episodes)
        density = target_density * ramp
        world.randomize_obstacles(density)   # 每 ep 都換新圖（受密度控制）
        world.reset()
        past_actions = []
        state = encode_state(world, past_actions)
        
        initial_score = world.score  # 記錄初始分數
        coins_collected = 0         # 本代已吃到的金幣數

        for step in range(max_steps_per_ep):
            steps_done += 1

            epsilon = 0.05 + 0.95 * math.exp(-steps_done / 1_000_000)

            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()

            base_reward, _ = world.step(action)

            # 檢查是否吃到金幣（score 增加表示吃到一枚）
            current_score = world.score
            new_coins = max(0, current_score - initial_score - coins_collected)
            if new_coins > 0:
                coins_collected += new_coins

            # 重新設計獎勵：
            # - 每步小懲罰，鼓勵更少步數
            # - 吃到金幣給小獎勵
            # - 第三枚金幣（達成目標）給與步數成反比的大獎勵
            reward = -0.01  # 每步小懲罰
            if new_coins > 0 and coins_collected < 3:
                reward += 1.0  # 中途吃到第1或第2枚的小獎勵
            if coins_collected >= 3:
                reward += 50.0 - step * 0.1  # 達成目標的大獎勵（步數越少越好）
            
            past_actions.append(action)
            if len(past_actions) > 6:
                past_actions.pop(0)
            next_state = encode_state(world, past_actions)

            done = coins_collected >= 3  # 吃到三個金幣就結束（不重生）
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= 256:
                batch = replay_buffer.sample(256)
                state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
                reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
                next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
                done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(DEVICE)

                current_q = q_net(state_batch).gather(1, action_batch)

                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                    target_q = reward_batch + 0.99 * next_q * (1 - done_batch)  # done時不計算未來獎勵

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 記錄loss
                loss_history.append(loss.item())

            if steps_done % 1000 == 0:
                target_net.load_state_dict(q_net.state_dict())
            
            # 吃到三個金幣就結束這一代
            if coins_collected >= 3:
                break

        steps_history.append(step + 1)  # 記錄本代用了幾步
        
        if (ep + 1) % 50 == 0:
            avg_steps = np.mean(steps_history[-50:])
            print(f"Ep {ep+1} Steps: {step+1}  Avg50: {avg_steps:.1f}  Coins: {coins_collected}")
        
        # 如果連續很多代都沒吃到，可能需要調整

    # 訓練完成，畫loss圖並存檔
    import matplotlib
    matplotlib.use('Agg')  # 無GUI後端
    import matplotlib.pyplot as plt
    
    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, alpha=0.3, label='Raw Loss')
        # 計算移動平均讓曲線更平滑
        window = min(100, len(loss_history) // 10)
        if window > 1:
            smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(loss_history)), smoothed, linewidth=2, label=f'MA-{window}')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('RL Agent Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('rl_training_loss.png', dpi=150)
        print(f"\n✓ Loss圖已存至: rl_training_loss.png")
        plt.close()

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

    # LSTM訓練完成，畫loss圖並存檔
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, linewidth=2, marker='o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Predictor Training Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('lstm_training_loss.png', dpi=150)
        print(f"\n✓ LSTM Loss圖已存至: lstm_training_loss.png")
        plt.close()

    return lstm
