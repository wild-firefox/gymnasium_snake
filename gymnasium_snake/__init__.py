from gymnasium.envs.registration import register
from gymnasium_snake.snake_env import SnakeEnv

register(
    id='Snake-v0',
    entry_point='gymnasium_snake:SnakeEnv',
    max_episode_steps=200,
)