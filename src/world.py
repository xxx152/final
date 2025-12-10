import random
import numpy as np
from .constants import GRID_W, GRID_H, NUM_REWARDS, REWARD_K, OBS_K, ACTION_HISTORY, ACTION_SPACE


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
        num_obs = int((self.w * self.h) * 0.05)
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
                    if (dx == 0 and dy == 0):
                        continue
                    check_x = ox + dx
                    check_y = oy + dy
                    if 0 <= check_x < self.w and 0 <= check_y < self.h:
                        is_neighbor = lambda x, y: any(o.x == x and o.y == y for o in self.obstacles)
                        neighbor1 = (ox + 1 - dx, oy - dy)
                        neighbor2 = (ox - dx, oy + 1 - dy)
                        if is_neighbor(check_x, check_y) and \
                           is_neighbor(neighbor1[0], neighbor1[1]) and \
                           is_neighbor(neighbor2[0], neighbor2[1]):
                            is_2x2_block = True
                            break
                if is_2x2_block:
                    break
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
