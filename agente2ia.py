import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import ale_py

# ===============================
# Red neuronal para DQN
# ===============================
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # entrada: 4 frames apilados
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x / 255.0  # normalización
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===============================
# Replay Buffer
# ===============================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ===============================
# Preprocesamiento de frames
# ===============================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized

def stack_frames(frames, new_frame):
    frames.append(new_frame)
    if len(frames) > 4:
        frames.pop(0)
    return np.stack(frames, axis=0)

# ===============================
# Entrenamiento con DQN
# ===============================
def train_dqn():
    env = gym.make("Amidar-v4", render_mode=None)
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(action_dim).to(device)
    target_net = DQN(action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer()

    episodes = 1000
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    update_target = 1000  # cada tantos pasos actualizamos red objetivo
    total_steps = 0

    for ep in range(episodes):
        state, _ = env.reset()
        frame = preprocess_frame(state)
        frames = [frame] * 4
        state = np.stack(frames, axis=0)

        done = False
        episode_reward = 0

        while not done:
            total_steps += 1

            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)

                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_frame = preprocess_frame(next_state)
            next_state_stacked = stack_frames(frames.copy(), next_frame)

            replay_buffer.push(state, action, reward, next_state_stacked, done)

            state = next_state_stacked
            episode_reward += reward

            # Entrenamiento
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)


                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Actualizar red objetivo
            if total_steps % update_target == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {ep}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

        # Guardar modelo cada 50 episodios
        if ep % 50 == 0:
            torch.save(policy_net.state_dict(), f"dqn_amidar_ep{ep}.pth")

    env.close()

if __name__ == "__main__":
    train_dqn()
