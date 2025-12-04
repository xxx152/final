import os
import pygame
from .constants import SCREEN_W, SCREEN_W_BASE, SCREEN_H, GRID_W, GRID_H, CELL, FPS, RTT_FRAMES


class Visualizer:
    def __init__(self, headless=None):
        # auto headless if no display
        if headless is None:
            headless = not bool(os.environ.get("DISPLAY"))
        self.headless = headless

        if self.headless:
            self.screen = None
            self.font = None
            self.big_font = None
            self.clock = None
            self.speed_mode = 0
        else:
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
                raise SystemExit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: self.speed_mode = 1
                elif event.key == pygame.K_2: self.speed_mode = 2
                elif event.key == pygame.K_3: self.speed_mode = 3
                elif event.key == pygame.K_4: self.speed_mode = 4
                elif event.key == pygame.K_0: self.speed_mode = 0

    def wait_frame(self):
        if self.headless:
            return
        if self.speed_mode == 1: self.clock.tick(5)
        elif self.speed_mode == 2: self.clock.tick(30)
        elif self.speed_mode == 3: self.clock.tick(120)

    def draw(self, world, phase_text, info_text, extra_info=""):
        if self.headless or self.speed_mode == 0:
            return

        game_area = pygame.Rect(0, 0, SCREEN_W_BASE, SCREEN_H)
        self.screen.fill((20, 20, 20), rect=game_area)

        for x in range(GRID_W):
            for y in range(GRID_H):
                pygame.draw.rect(self.screen, (40, 40, 40), (x*CELL, y*CELL, CELL, CELL), 1)

        for o in world.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (o.x*CELL+4, o.y*CELL+4, CELL-8, CELL-8))

        for r in world.rewards:
            pygame.draw.circle(self.screen, (255, 215, 0), (r.x*CELL+CELL//2, r.y*CELL+CELL//2), CELL//3)

        if "RL" in phase_text or "Data" in phase_text:
            pygame.draw.circle(self.screen, (255, 100, 100), (world.ax*CELL+CELL//2, world.ay*CELL+CELL//2), CELL//3)
        else:
            pygame.draw.circle(self.screen, (0, 180, 0), (world.ax*CELL+CELL//2, world.ay*CELL+CELL//2), CELL//3)
            pygame.draw.circle(self.screen, (50, 150, 255), (world.cx*CELL+CELL//2, world.cy*CELL+CELL//2), CELL//4)

        lines = [
            f"Phase: {phase_text} (Fixed RL)",
            f"Speed: [1]Slow [2]Norm [3]Fast [4]Max [0]Skip",
            info_text,
            extra_info,
        ]
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
        if self.headless or self.speed_mode == 0:
            return
        # For brevity, reuse draw() for now with summarized text
        info = f"Ep {ep+1}/{total_episodes} Eps={epsilon:.2f} Score={world.score} R={current_reward:.1f}"
        self.draw(world, "1. RL Training", info, "")
