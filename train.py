import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import csv

from model import DQN
from flappy_env import FlappyBirdEnv

# Hyperparameters
EPISODES = 2000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

env = FlappyBirdEnv()
state_dim = 5
action_dim = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

memory = deque(maxlen=MEMORY_SIZE)
episode_rewards = []
episode_scores = []

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state)
    return q_values.argmax().item()

def train_step():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = F.mse_loss(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon = EPSILON_START
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    while True:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train_step()
        if done:
            break

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    episode_rewards.append(total_reward)
    episode_scores.append(env.score)

    # Ghi log
    with open("flappy_training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, env.score, total_reward])

    print(f"Episode {episode}, Score: {env.score}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# Lưu mô hình
torch.save(policy_net.state_dict(), "dqn_flappy.pt")

# Vẽ đồ thị
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label='Total Reward')
plt.plot(episode_scores, label='Score')
plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Training Progress")
plt.legend()
plt.grid()
plt.savefig("training_plot.png")
plt.show()
