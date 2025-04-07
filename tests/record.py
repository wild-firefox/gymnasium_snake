import gymnasium as gym
import gymnasium_snake
import time
import random

def main():
    env = gym.make('Snake-v0', render_mode="rgb_array")
    # 使用RecordVideo录制
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder="./video",
        episode_trigger=lambda episode_id: True  # 录制每一个episode
    )
    
    
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
        time.sleep(0.1)
        
        total_reward += reward
        step_count += 1
        print(f'Step {step_count}, reward = {reward}, terminated = {terminated}, truncated = {truncated}')
    
    print(f'Total reward: {total_reward}')
    env.close()

if __name__ == "__main__":
    main()