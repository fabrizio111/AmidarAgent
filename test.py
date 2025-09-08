import gymnasium as gym
import ale_py

env = gym.make("ALE/Amidar-v5", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
