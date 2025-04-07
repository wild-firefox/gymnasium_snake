import gymnasium as gym
import gymnasium_snake
import time
import random

def main():
    env = gym.make('Snake-v0', render_mode="human")
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        # 确保选择有效动作
        action_mask = info.get("action_mask", [True] * 4)
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if valid_actions:
            action = random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)
        
        total_reward += reward
        step_count += 1
        print(f'Step {step_count}, reward = {reward}, terminated = {terminated}, truncated = {truncated}')
    
    print(f'Total reward: {total_reward}')
    env.close()

if __name__ == "__main__":
    main()