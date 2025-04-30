import torch, torch.nn as nn, torch.optim as optim
import random, numpy as np

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state)
        return torch.argmax(q).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))
        if len(self.memory) > 5000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_vals = self.model(states).gather(1, actions)
        q_next = self.target(next_states).max(1)[0].unsqueeze(1).detach()
        q_target = rewards + self.gamma * q_next * (~dones)

        loss = self.criterion(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
