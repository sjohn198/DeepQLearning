import gymnasium as gym
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np

env_train = gym.make("CartPole-v1")
env_test = gym.make("CartPole-v1", render_mode="human")

class QPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
total_episodes = 10000
epsilon = 1
epsilone_decay_rate = 1 / total_episodes

learning_rate = 0.1
discount_factor = 0.001

batch_size = 64
rp = ReplayBuffer(10000)

policy_network = QPolicyNetwork()
target_network = QPolicyNetwork()
target_network.load_state_dict(policy_network.state_dict())

optimizer = torch.optim.Adam(policy_network.parameters(), lr = learning_rate)
loss = nn.MSELoss()

for episode in range(total_episodes):
    state, info = env_train.reset()
    done = False
    truncated = False

    while not (done or truncated):
        if np.random.uniform(0,1) > epsilon:
            q_values = policy_network.predict(state)
            action = np.argmax(q_values[0])
        else:
            action = env_train.action_space.sample()

        next_state, reward, done, truncated, info = env_train.step(action)

        rp.push(state, action, next_state, reward, done, truncated)

        state = next_state

        if len(rp) < batch_size:
            continue

        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch, truncateds_batch = rp.sample(batch_size)

        future_q_values = target_network.predict(next_states_batch)
        max_future_q = np.max(future_q_values, axis=1)

        if max_future_q == 0:
            break

