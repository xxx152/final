# Grid-based Zero-Latency Demo (Fixed RL + LSTM)

import pygame
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
import math
from collections import deque, namedtuple

# GPU 設備偵測
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 參數設定 ---
GRID_W, GRID_H = 16, 12
CELL = 48
SCREEN_W_BASE = GRID_W * CELL
SCREEN_W = SCREEN_W_BASE + 300
SCREEN_H = GRID_H * CELL

FPS = 60
RTT_FRAMES = 15
MOVE_COOLDOWN = 5
NUM_REWARDS = 5

NUM_ACTIONS = 5
NOOP = 4
ACTION_SPACE = NUM_ACTIONS
ACTION_HISTORY = 6
SEQ_LEN = 8

# 新增：狀態維度參數
REWARD_K = 5    # 看到全部金幣
OBS_K = 20
BASE_FEATURE_DIM = 2 + 2 * REWARD_K + 2 * OBS_K    # 2 + 10 + 40 = 52
INPUT_DIM = BASE_FEATURE_DIM + (ACTION_HISTORY * ACTION_SPACE)  # 52 + 30 = 82

# 訓練參數（大幅強化版）
RL_EPISODES = 3000
RL_STEPS_PER_EP = 750         # 步數增加，代理有更多時間收集
DATA_COLLECTION_SIZE = 200000  # 更多專家數據
LSTM_TRAIN_EPOCHS = 80
MAP_CHANGE_FREQ = 150
BATCH_SIZE = 256
REPLAY_CAPACITY = 150000
TARGET_UPDATE_FREQ = 800      # 步數
GAMMA = 0.99
LR = 3e-4
PROGRESS_SCALE = 1.0           # 關鍵：讓靠近金幣的動作淨收益為正

# --- 繪圖工具 ---
class Visualizer:
    def __init__(self, headless=None):
        # 決定是否為無頭模式（無顯示環境）
        if headless is None:
            # 若沒有 DISPLAY（常見於 Linux server）則啟用無頭
            headless = not bool(os.environ.get("DISPLAY"))

        self.headless = headless

        if self.headless:
            # 無頭模式：不初始化視窗與字型，繪圖皆為 no-op
            self.screen = None
            self.font = None
            self.big_font = None
            self.clock = None
            self.speed_mode = 0  # 直接跳過所有繪圖
        else:
            # 一般模式：初始化 pygame 與視窗
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Zero Latency: Multi-Coin Demo")
            self.font = pygame.font.SysFont("Arial", 18)
            self.big_font = pygame.font.SysFont("Arial", 24, bold=True)
            self.clock = pygame.time.Clock()
            self.speed_mode = 3 

    def handle_speed_input(self):
        if self.headless:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: self.speed_mode = 1
                elif event.key == pygame.K_2: self.speed_mode = 2
                elif event.key == pygame.K_3: self.speed_mode = 3
                elif event.key == pygame.K_4: self.speed_mode = 4 # Max FPS (繪圖但不鎖定幀數)
                elif event.key == pygame.K_0: self.speed_mode = 0 

    def wait_frame(self):
        if self.headless:
            return
        if self.speed_mode == 1: self.clock.tick(5)
        elif self.speed_mode == 2: self.clock.tick(30)
        elif self.speed_mode == 3: self.clock.tick(120)

    def draw(self, world, phase_text, info_text, extra_info=""):
        if self.headless or self.speed_mode == 0: return

        game_area = pygame.Rect(0, 0, SCREEN_W_BASE, SCREEN_H)
        self.screen.fill((20, 20, 20), rect=game_area) 

        for x in range(GRID_W):
            for y in range(GRID_H):
                pygame.draw.rect(self.screen, (40, 40, 40), (x*CELL, y*CELL, CELL, CELL), 1)

        for o in world.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (o.x*CELL+4, o.y*CELL+4, CELL-8, CELL-8))

        for r in world.rewards:
            color = (255, 215, 0)
            pygame.draw.circle(self.screen, color, (r.x*CELL+CELL//2, r.y*CELL+CELL//2), CELL//3)

        if "RL" in phase_text or "Data" in phase_text:
            pygame.draw.circle(self.screen, (255, 100, 100), (world.ax*CELL+CELL//2, world.ay*CELL+CELL//2), CELL//3)
        else:
            pygame.draw.circle(self.screen, (0, 180, 0), (world.ax*CELL+CELL//2, world.ay*CELL+CELL//2), CELL//3)
            pygame.draw.circle(self.screen, (50, 150, 255), (world.cx*CELL+CELL//2, world.cy*CELL+CELL//2), CELL//4)

        lines = [f"Phase: {phase_text} (Fixed RL)", f"Speed: [1]Slow [2]Norm [3]Fast [4]Max [0]Skip", info_text, extra_info]
        y = 10
        for line in lines:
            s = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(s, (10, y))
            y += 25
            
        latency_ms = (RTT_FRAMES / FPS) * 1000
        lat_text = self.big_font.render(f"Latency: {int(latency_ms)} ms", True, (255, 50, 50))
        self.screen.blit(lat_text, (SCREEN_W_BASE - 200, 10))

        pygame.display.flip()

    def draw_loss_graph(self, epoch, current_loss, loss_history, total_epochs):
        if self.headless:
            return
        self.screen.fill((20, 20, 20))
        
        PLOT_X = 50           
        PLOT_Y = 50           
        PLOT_W = SCREEN_W - 100 
        PLOT_H = SCREEN_H - 100 
        
        pygame.draw.rect(self.screen, (30, 30, 30), (PLOT_X, PLOT_Y, PLOT_W, PLOT_H))
        pygame.draw.line(self.screen, (150, 150, 150), (PLOT_X, PLOT_Y + PLOT_H), (PLOT_X + PLOT_W, PLOT_Y + PLOT_H), 2)
        pygame.draw.line(self.screen, (150, 150, 150), (PLOT_X, PLOT_Y), (PLOT_X, PLOT_Y + PLOT_H), 2)
        
        if len(loss_history) > 1:
            max_loss = max(loss_history) * 1.05 
            max_epochs_to_show = 100 
            history_to_draw = loss_history[-max_epochs_to_show:]
            
            point_list = []
            scale_w = PLOT_W / (len(history_to_draw) - 1) if len(history_to_draw) > 1 else PLOT_W
            
            for i, loss in enumerate(history_to_draw):
                x = PLOT_X + i * scale_w / len(history_to_draw)
                y_norm = 1 - (loss / max_loss) 
                y = PLOT_Y + y_norm * PLOT_H 
                point_list.append((x, y))
            
            if len(point_list) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, point_list, 2)
            
            s_title = self.big_font.render("Phase 2b: LSTM Training Loss", True, (255, 255, 255))
            self.screen.blit(s_title, (PLOT_X, 10))
            
            s_loss = self.font.render(f"Current Loss: {current_loss:.4f}", True, (255, 255, 255))
            self.screen.blit(s_loss, (PLOT_X, PLOT_Y + PLOT_H + 5))
            
            s_epoch = self.font.render(f"Epoch: {epoch+1}/{total_epochs}", True, (255, 255, 255))
            self.screen.blit(s_epoch, (PLOT_X + PLOT_W - 150, PLOT_Y + PLOT_H + 5))
            
            s_max = self.font.render(f"Max Loss: {max_loss:.2f}", True, (255, 255, 255))
            self.screen.blit(s_max, (PLOT_X + 5, PLOT_Y + 5))
            
        pygame.display.flip()

    def draw_rl_status(self, world, ep, score_history, total_episodes, epsilon, map_change_freq, current_reward):
        if self.headless or self.speed_mode == 0: return

        game_area = pygame.Rect(0, 0, SCREEN_W_BASE, SCREEN_H)
        self.screen.fill((20, 20, 20), rect=game_area) 

        for x in range(GRID_W):
            for y in range(GRID_H):
                pygame.draw.rect(self.screen, (40, 40, 40), (x*CELL, y*CELL, CELL, CELL), 1)

        for o in world.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (o.x*CELL+4, o.y*CELL+4, CELL-8, CELL-8))

        for r in world.rewards:
            color = (255, 215, 0)
            pygame.draw.circle(self.screen, color, (r.x*CELL+CELL//2, r.y*CELL+CELL//2), CELL//3)

        pygame.draw.circle(self.screen, (255, 100, 100), (world.ax*CELL+CELL//2, world.ay*CELL+CELL//2), CELL//3)

        lines = [
        f"Phase: 1. RL Training", 
        f"Speed: [1]Slow [2]Norm [3]Fast [4]Max FPS [0]Skip",
        f"Ep: {ep+1}/{total_episodes} (Map Change in {map_change_freq - ep%map_change_freq})",
        f"Eps: {epsilon:.2f}",
        f"Current Score: {world.score}",
        f"Reward: {current_reward:.2f}" 
        ]
        y = 10
        for line in lines:
            s = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(s, (10, y))
            y += 25
            
        latency_ms = (RTT_FRAMES / FPS) * 1000
        lat_text = self.big_font.render(f"Latency: {int(latency_ms)} ms", True, (255, 50, 50))
        self.screen.blit(lat_text, (SCREEN_W_BASE - 200, 10))

        PLOT_X = SCREEN_W_BASE + 20
        PLOT_Y = 50
        PLOT_W = 260
        PLOT_H = SCREEN_H - 100
        
        plot_rect = pygame.Rect(SCREEN_W_BASE, 0, SCREEN_W - SCREEN_W_BASE, SCREEN_H)
        self.screen.fill((10, 10, 10), rect=plot_rect)

        pygame.draw.rect(self.screen, (30, 30, 30), (PLOT_X, PLOT_Y, PLOT_W, PLOT_H))
        pygame.draw.line(self.screen, (150, 150, 150), (PLOT_X, PLOT_Y + PLOT_H), (PLOT_X + PLOT_W, PLOT_Y + PLOT_H), 2)
        pygame.draw.line(self.screen, (150, 150, 150), (PLOT_X, PLOT_Y), (PLOT_X, PLOT_Y + PLOT_H), 2)
        
        if len(score_history) > 1:
            max_score = max(score_history) if score_history else 1
            min_score = min(score_history) if score_history else 0
            score_range = max(1.0, max_score - min_score)
            
            max_episodes_to_show = 200 
            history_to_draw = score_history[-max_episodes_to_show:]
            N = len(history_to_draw)
            
            point_list = []
            
            if N > 1:
                step_x = PLOT_W / (N - 1) 
            else:
                step_x = 0 
            
            for i, score in enumerate(history_to_draw):
                if N > 1:
                    x = PLOT_X + i * step_x 
                else:
                    x = PLOT_X + PLOT_W / 2 
                
                y_norm = 1 - ((score - min_score) / score_range) 
                y = PLOT_Y + y_norm * PLOT_H 
                point_list.append((x, y))
            
            if len(point_list) > 1:
                pygame.draw.lines(self.screen, (255, 50, 50), False, point_list, 2)
            
            s_title = self.big_font.render("Episode Score Trend", True, (255, 255, 255))
            self.screen.blit(s_title, (PLOT_X, 10))
            
            s_max = self.font.render(f"Max: {max_score:.1f}", True, (255, 255, 255))
            self.screen.blit(s_max, (PLOT_X + 5, PLOT_Y + 5))
            
            s_min = self.font.render(f"Min: {min_score:.1f}", True, (255, 255, 255))
            self.screen.blit(s_min, (PLOT_X + 5, PLOT_Y + PLOT_H - 20))
            
        pygame.display.flip()

# --- 環境邏輯 ---
class GridWorld:
    def __init__(self, w=GRID_W, h=GRID_H):
        self.w, self.h = w, h
        self.ax, self.ay = w // 2, h // 2
        self.cx, self.cy = self.ax, self.ay
        self.rewards = []
        self.obstacles = []
        self.score = 0
        self.randomize_obstacles() 

    def randomize_obstacles(self):
        self.obstacles = []
        num_obs = int((self.w * self.h) * 0.15) 
        
        count = 0
        attempts = 0
        while count < num_obs and attempts < 1000:
            ox = random.randint(0, self.w - 1)
            oy = random.randint(0, self.h - 1)
            
            if abs(ox - self.w//2) <= 1 and abs(oy - self.h//2) <= 1:
                attempts += 1
                continue
            
            if any(o.x == ox and o.y == oy for o in self.obstacles):
                attempts += 1
                continue
                
            is_2x2_block = False
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    if (dx == 0 and dy == 0): continue
                    
                    check_x = ox + dx
                    check_y = oy + dy
                    
                    if 0 <= check_x < self.w and 0 <= check_y < self.h:
                        
                        is_neighbor_obstacle = lambda x, y: any(o.x == x and o.y == y for o in self.obstacles)
                        
                        neighbor1 = (ox + 1 - dx, oy - dy)
                        neighbor2 = (ox - dx, oy + 1 - dy)
                        
                        if is_neighbor_obstacle(check_x, check_y) and \
                           is_neighbor_obstacle(neighbor1[0], neighbor1[1]) and \
                           is_neighbor_obstacle(neighbor2[0], neighbor2[1]):
                           is_2x2_block = True
                           break
                if is_2x2_block: break

            if is_2x2_block:
                attempts += 1
                continue

            self.obstacles.append(type('Obj', (object,), {'x': ox, 'y': oy}))
            count += 1

    def spawn_reward(self):
        attempts = 0
        while attempts < 500:
            rx = random.randint(0, self.w - 1)
            ry = random.randint(0, self.h - 1)
            
            if (rx == self.ax and ry == self.ay) or \
               any(o.x == rx and o.y == ry for o in self.obstacles) or \
               any(r.x == rx and r.y == ry for r in self.rewards):
                attempts += 1
                continue
                
            empty_neighbor_count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = rx + dx, ry + dy
                
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    if not any(o.x == nx and o.y == ny for o in self.obstacles):
                        empty_neighbor_count += 1
            
            if empty_neighbor_count < 1:
                attempts += 1
                continue
            
            new_r = type('Obj', (object,), {'x': rx, 'y': ry, 'collected': False})
            self.rewards.append(new_r)
            return

        print("Warning: Could not find an accessible spot for a new reward. Map may be too complex.")

    def reset(self):
        self.ax, self.ay = self.w // 2, self.h // 2
        self.cx, self.cy = self.ax, self.ay
        self.score = 0
        self.rewards = []
        for _ in range(NUM_REWARDS):
            self.spawn_reward()
        return self

    def get_k_nearest_vecs(self, objects, k=3):
        valid_objs = [o for o in objects if not getattr(o, 'collected', False)]
        dist_list = []
        for o in valid_objs:
            dist = abs(o.x - self.ax) + abs(o.y - self.ay)
            dist_list.append((dist, o))
        dist_list.sort(key=lambda x: x[0])
        vecs = []
        for i in range(k):
            if i < len(dist_list):
                obj = dist_list[i][1]
                dx = (obj.x - self.ax) / max(1, self.w - 1)
                dy = (obj.y - self.ay) / max(1, self.h - 1)
                vecs.extend([dx, dy])
            else:
                vecs.extend([0.0, 0.0])
        return vecs

    def step(self, action):
        nx, ny = self.ax, self.ay
        if action == 0: ny -= 1
        elif action == 1: ny += 1
        elif action == 2: nx -= 1
        elif action == 3: nx += 1

        reward = -0.1
        done = False
        
        if not (0 <= nx < self.w and 0 <= ny < self.h) or \
           any(o.x == nx and o.y == ny for o in self.obstacles):
            reward = -1.0
        else:
            self.ax, self.ay = nx, ny
            
        hit_reward = None
        for r in self.rewards:
            if r.x == self.ax and r.y == self.ay:
                hit_reward = r
                break
        
        if hit_reward:
            self.rewards.remove(hit_reward)
            reward = 10.0
            self.score += 1
            self.spawn_reward()
            
        return reward, done

    def client_apply(self, action):
        nx, ny = self.cx, self.cy
        if action == 0: ny -= 1
        elif action == 1: ny += 1
        elif action == 2: nx -= 1
        elif action == 3: nx += 1
        if (0 <= nx < self.w and 0 <= ny < self.h) and \
           not any(o.x == nx and o.y == ny for o in self.obstacles):
            self.cx, self.cy = nx, ny

def encode_state(world, past_actions=None):
    if past_actions is None:
        past_actions = []
    ax = world.ax / (world.w - 1)
    ay = world.ay / (world.h - 1)
    r_vec = world.get_k_nearest_vecs(world.rewards, k=REWARD_K)
    o_vec = world.get_k_nearest_vecs(world.obstacles, k=OBS_K)
    features = [ax, ay] + r_vec + o_vec
    
    action_part = []
    for i in range(ACTION_HISTORY):
        if i < len(past_actions):
            action_part.extend([1 if x == past_actions[-1-i] else 0 for x in range(ACTION_SPACE)])
        else:
            action_part.extend([0] * ACTION_SPACE)
    return np.array(features + action_part, dtype=np.float32)

# --- 模型（加大 + 使用完整狀態）---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.fc(x)                     # 使用完整狀態（包含 past actions）

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_actions=NUM_ACTIONS):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# --- 真正修正的 RL 訓練（DQN + Replay + Target Net + 優秀 reward shaping）---
def train_rl_agent(viz):
    world = GridWorld()
    q_net = QNetwork(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
    target_net = QNetwork(INPUT_DIM, NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=2e-4)  # 2e-4 是目前最穩的甜點
    replay_buffer = ReplayBuffer(200000)

    score_history = []
    steps_done = 0

    # === 超強 warm-up：30000 筆 + 每 80 步強制 reset，避免單 episode 太長污染 ===
    print("Strong Warm-up replay buffer (30000 steps)...")
    past_actions_warm = []
    for _ in range(30000):
        if _ % 80 == 0:
            world.reset()
            past_actions_warm = []
        action = random.randint(0, NUM_ACTIONS - 1)
        reward, _ = world.step(action)
        # maintain a short history during warm-up for consistency
        past_actions_warm.append(action)
        if len(past_actions_warm) > ACTION_HISTORY:
            past_actions_warm.pop(0)
        next_state = encode_state(world, past_actions_warm)
        replay_buffer.push(encode_state(world, []), action, reward, next_state, False)

    past_actions = []

    for ep in range(RL_EPISODES):
        if ep % 150 == 0 and ep > 0:  # 150 ep 換一次圖，最穩
            world.randomize_obstacles()
            
        world.reset()
        past_actions = []
        state = encode_state(world, past_actions)

        for step in range(800):  # 絕對不能超過 800！800 是神級邊界，超過必死
            steps_done += 1

            # 超慢探索，後期仍有 10% 左右隨機動作防止過擬合
            epsilon = 0.05 + 0.95 * math.exp(-steps_done / 1200000)

            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    action = q_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax().item()

            # === PBRS（理論保證不改變最優策略，但給超強引導）===
            dists_old = [abs(r.x - world.ax) + abs(r.y - world.ay) for r in world.rewards]
            min_dist_old = min(dists_old) if dists_old else 0
            old_potential = -min_dist_old

            base_reward, _ = world.step(action)

            dists_new = [abs(r.x - world.ax) + abs(r.y - world.ay) for r in world.rewards]
            min_dist_new = min(dists_new) if dists_new else 0
            new_potential = -min_dist_new

            shaping = GAMMA * new_potential - old_potential   # PBRS 公式
            reward = base_reward + shaping

            past_actions.append(action)
            if len(past_actions) > ACTION_HISTORY:
                past_actions.pop(0)
            next_state = encode_state(world, past_actions)

            replay_buffer.push(state, action, reward, next_state, step == 799)

            state = next_state

            # === 訓練：DDQN + Huber Loss ===
            if len(replay_buffer) >= BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                # sample already returns a Transition of tuples; use it directly
                batch = transitions

                state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
                action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
                reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
                next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
                done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(DEVICE)

                current_q = q_net(state_batch).gather(1, action_batch)

                with torch.no_grad():
                    next_actions = q_net(next_state_batch).argmax(1).unsqueeze(1)
                    next_q = target_net(next_state_batch).gather(1, next_actions)
                    target_q = reward_batch + 0.99 * next_q * (1 - done_batch)

                loss = nn.SmoothL1Loss()(current_q, target_q)  # Huber 超穩

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

            # target net 每 1500 步更新一次（最穩）
            if steps_done % 1500 == 0:
                target_net.load_state_dict(q_net.state_dict())

            if viz.speed_mode != 0:
                viz.draw_rl_status(world, ep, score_history, RL_EPISODES, epsilon, 150, reward)
                if viz.speed_mode != 4:
                    viz.wait_frame()

        score_history.append(world.score)

        if (ep + 1) % 50 == 0:
            avg50 = np.mean(score_history[-50:])
            print(f"Ep {ep+1} Score: {world.score}  Avg50: {avg50:.1f}")

    return q_net

# --- LSTM 數據收集（也加入 past_actions，讓狀態一致）---
def train_lstm_pipeline(viz, rl_agent):
    world = GridWorld()
    inputs, targets = [], []
    
    world.reset()
    past_actions = []
    state_hist = [encode_state(world, past_actions) for _ in range(SEQ_LEN)]
    
    steps = 0
    map_timer = 0

    while steps < DATA_COLLECTION_SIZE:
        viz.handle_speed_input()
        if viz.speed_mode != 0:
            viz.draw(world, "2a. Gen Data (Fixed)", f"Steps: {steps}/{DATA_COLLECTION_SIZE}", "Collecting expert data...")
            if viz.speed_mode != 4:
                viz.wait_frame()

        curr_vec = state_hist[-1]
        with torch.no_grad():
            q_vals = rl_agent(torch.FloatTensor(curr_vec).unsqueeze(0).to(DEVICE))
            expert_action = torch.argmax(q_vals).item()

        # 加入少量噪聲（DAgger 風格）
        action = expert_action if random.random() < 0.9 else random.randint(0, NUM_ACTIONS - 1)

        world.step(action)

        past_actions.append(action)
        if len(past_actions) > ACTION_HISTORY:
            past_actions.pop(0)

        next_state = encode_state(world, past_actions)
        state_hist.append(next_state)
        state_hist.pop(0)

        inputs.append(np.array(state_hist[-SEQ_LEN:]))
        targets.append(expert_action)      # 標籤永遠是專家動作

        steps += 1
        map_timer += 1
        if map_timer > 600:
            world.randomize_obstacles()
            world.reset()
            past_actions = []
            state_hist = [encode_state(world, past_actions) for _ in range(SEQ_LEN)]
            map_timer = 0

    # LSTM 訓練（不變，只是數據品質大幅提升）
    lstm = LSTMPredictor(INPUT_DIM).to(DEVICE)
    opt = optim.Adam(lstm.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss().to(DEVICE)

    ds = torch.utils.data.TensorDataset(torch.FloatTensor(np.array(inputs)), torch.LongTensor(targets))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    loss_history = []
    for epoch in range(LSTM_TRAIN_EPOCHS):
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

        if viz.speed_mode != 0:
            viz.draw_loss_graph(epoch, avg_loss, loss_history, LSTM_TRAIN_EPOCHS)
            if viz.speed_mode != 4:
                viz.wait_frame()

    return lstm

def run_game(viz, lstm_model):
    world = GridWorld().reset()
    state_hist = []
    past_actions = []
    delayed_u_actions = []
    server_send_buf = []
    
    CLIENT_ONE_WAY = RTT_FRAMES // 2
    for _ in range(SEQ_LEN): state_hist.append(encode_state(world, []))
    
    viz.speed_mode = 2
    running = True
    frame = 0
    total_preds, correct_preds = 0, 0
    input_cooldown_timer = 0 
    
    while running:
        frame += 1
        viz.handle_speed_input()
        
        keys = pygame.key.get_pressed()
        ua = None
        if input_cooldown_timer <= 0:
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
        if len(past_actions) > ACTION_HISTORY: past_actions.pop(0)
        delayed_u_actions.append([CLIENT_ONE_WAY, curr_act])
        
        # Client
        for buf in server_send_buf: buf[0] -= 1
        arrived = [b for b in server_send_buf if b[0] <= 0]
        server_send_buf = [b for b in server_send_buf if b[0] > 0]
        latest_pred = arrived[-1] if arrived else None
        
        match = False
        if latest_pred and curr_act != NOOP: 
            total_preds += 1
            if curr_act in latest_pred[1]:
                correct_preds += 1
                match = True
        
        if curr_act != NOOP: world.client_apply(curr_act)
        
        # Server
        for item in delayed_u_actions: item[0] -= 1
        arrived_acts = [a for a in delayed_u_actions if a[0] <= 0]
        delayed_u_actions = [a for a in delayed_u_actions if a[0] > 0]
        if arrived_acts:
            s_act = arrived_acts[-1][1]
            if s_act != NOOP: world.step(s_act)
            
        # Predict
        curr_enc = encode_state(world, past_actions)
        state_hist.append(curr_enc)
        if len(state_hist) > 200: state_hist.pop(0)
        
        seq = torch.tensor([state_hist[-SEQ_LEN:]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(lstm_model(seq), dim=1)[0].cpu().numpy()
        top2 = probs.argsort()[-2:][::-1].tolist()
        server_send_buf.append([CLIENT_ONE_WAY, top2])
        
        if frame % (FPS * 5) == 0: world.cx, world.cy = world.ax, world.ay
        
        acc = (correct_preds / total_preds * 100) if total_preds > 0 else 0.0
        viz.draw(world, "3. Play", f"Acc: {acc:.1f}%", f"Match: {match}")
        if viz.speed_mode != 4: 
            viz.wait_frame()

if __name__ == "__main__":
    viz = Visualizer(headless=None)  # None 自動偵測無頭模式
    print("Start Phase 1... (Training RL Agent)")
    viz.speed_mode = 3 
    rl_net = train_rl_agent(viz)

    print("Start Phase 2... (Training LSTM)")
    viz.speed_mode = 3
    lstm_net = train_lstm_pipeline(viz, rl_net)

    # 無頭模式：不開啟遊戲視窗，改存模型參數
    if viz.headless:
        timestamp = int(time.time())
        rl_path = f"rl_net_{timestamp}.pth"
        lstm_path = f"lstm_net_{timestamp}.pth"
        meta_path = f"training_meta_{timestamp}.json"
        torch.save(rl_net.state_dict(), rl_path)
        torch.save(lstm_net.state_dict(), lstm_path)
        meta = {
            "timestamp": timestamp,
            "device": str(DEVICE),
            "grid": {"w": GRID_W, "h": GRID_H},
            "cell": CELL,
            "num_rewards": NUM_REWARDS,
            "rtt_frames": RTT_FRAMES,
            "fps": FPS,
            "model": {
                "input_dim": INPUT_DIM,
                "action_space": ACTION_SPACE,
                "action_history": ACTION_HISTORY,
                "seq_len": SEQ_LEN
            },
            "training": {
                "rl_episodes": RL_EPISODES,
                "rl_steps_per_episode": RL_STEPS_PER_EP,
                "lstm_epochs": LSTM_TRAIN_EPOCHS,
                "replay_capacity": REPLAY_CAPACITY,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "lr": LR
            },
            "artifacts": {"rl": rl_path, "lstm": lstm_path}
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Headless mode: saved models to {rl_path}, {lstm_path} and metadata to {meta_path}.")
    else:
        print("Start Game... (Human vs AI)")
        run_game(viz, lstm_model=lstm_net)