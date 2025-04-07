import gymnasium as gym
import numpy as np
import random
import pygame
import sys
import time

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, width=20, height=10, render_mode=None):
        super(SnakeEnv, self).__init__()

        # 存储渲染模式
        self.render_mode = render_mode

        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.snake = None
        self.food = None
        self.action_mask = [True] * 4
        self.episode_length = 0
        self.episode_reward = 0
        self.episode = 0
        self.max_episode_length = 200
        self.cell_size = 20
        self.window_size = (self.width * self.cell_size,
                            self.height * self.cell_size)
        # 定义颜色
        self.color_bg = (255, 255, 255)
        self.color_head = (0, 120, 120)
        self.color_body = (0, 255, 0)
        self.color_food = (255, 0, 0)

        # 定义贪吃蛇环境的观测空间和行动空间
        low = np.array([-10, -1.0, 0, 0, 0, 0])  # 连续状态空间的最小值
        high = np.array([10, 1.0, 1.0, 1.0, 1.0, 1.0])  # 连续状态空间的最大值
        self.observation_space = gym.spaces.Box(low, high, shape=(6,), dtype=np.float32)

        # 0 左 1上 2右 3下
        self.action_space = gym.spaces.Discrete(4)
        
        # 为了支持动作掩码功能
        self.action_mask_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.int8)
        
        # 初始化pygame相关
        self.window = None
        self.font = None

    def generate_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.height, self.width))
        self.snake = [(0, 0)]
        self.food = self.generate_food()
        self.time = 1
        self.action_mask = self.get_mask()
        self.episode_length = 0
        self.episode_reward = 0
        self.update_grid()
        
        return self.get_state(), {"action_mask": self.action_mask}

    def get_state(self):
        state = []
        x, y = self.snake[0]
        fx, fy = self.food
        state.append(fx - x)
        state.append(fy - y)
        for gx, gy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                state.append(0)
                continue
            else:
                state.append(1)  # 四个方向可以走
        return np.array(state, dtype=np.float32)

    def update_grid(self):
        self.grid = np.zeros((self.height, self.width))
        x, y = self.snake[0]
        self.grid[y, x] = 1
        for x, y in self.snake[1:]:
            self.grid[y, x] = 2
        fx, fy = self.food
        self.grid[fy, fx] = 3

    def get_mask(self):
        action_mask = [True] * 4
        x, y = self.snake[0]
        for i, (gx, gy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                action_mask[i] = False
            else:
                action_mask[i] = True  # True则表示动作可以执行
        return action_mask

    def step(self, action):
        x, y = self.snake[0]
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x = x + direction[action][0]
        y = y + direction[action][1]

        self.episode_length += 1
        
        terminated = False
        truncated = False
        
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake:
            reward = -1
            terminated = True  # 游戏结束，碰到墙或自己
        elif (x, y) == self.food:
            reward = 1
            self.snake.insert(0, (x, y))
            self.food = self.generate_food()
            self.update_grid()
        else:
            fx, fy = self.food
            d = (abs(x - fx) + abs(y - fy))
            reward = 0
            self.snake.insert(0, (x, y))
            self.snake.pop()
            self.update_grid()
            
        if self.episode_length >= self.max_episode_length:
            truncated = True

        info = {}
        self.episode_reward += reward
        
        self.action_mask = self.get_mask()
        info["action_mask"] = self.action_mask
        
        if terminated or truncated:
            self.episode += 1
            details = {}
            details['r'] = self.episode_reward
            details['l'] = self.episode_length
            details['e'] = self.episode
            info['episode'] = details
            
        return self.get_state(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return
            
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.Font(None, 30)
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Snake Game")

        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.color_bg)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.grid[y, x]
                cell_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                        self.cell_size, self.cell_size)
                if cell_value == 0:  # 空白格子
                    pygame.draw.rect(canvas, (255, 255, 0), cell_rect, 1)
                elif cell_value == 1:  # 贪吃蛇头部
                    pygame.draw.rect(canvas, self.color_head, cell_rect)
                elif cell_value == 2:  # 贪吃蛇身体
                    pygame.draw.rect(canvas, self.color_body, cell_rect)
                elif cell_value == 3:  # 食物
                    pygame.draw.circle(canvas, self.color_food,
                                     (cell_rect.x + self.cell_size // 2, cell_rect.y + self.cell_size // 2),
                                     self.cell_size // 2)

        snake_length_text = self.font.render("Length: " + str(len(self.snake)),
                                            True, (0, 25, 25))
        canvas.blit(snake_length_text, (0, 0))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
        
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(canvas).transpose((1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None