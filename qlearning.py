import gymnasium as gym
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
    
device = get_device()

total_episodes = 10000
epsilon = 1
epsilone_decay_rate = 1 / total_episodes

learning_rate = 0.1
discount_factor = 0.001

batch_size = 64
rp = ReplayBuffer(10000)

policy_network = QPolicyNetwork().to(device)
target_network = QPolicyNetwork().to(device)
target_network.load_state_dict(policy_network.state_dict())

optimizer = torch.optim.Adam(policy_network.parameters(), lr = learning_rate)
loss_fn = nn.MSELoss()

average_losses = []

for episode in range(total_episodes):
    state, info = env_train.reset()
    done = False
    truncated = False

    current_episode_losses = []

    while not (done or truncated):
        if np.random.uniform(0,1) > epsilon:
            q_values = policy_network(state)
            action = np.argmax(q_values[0])
        else:
            action = env_train.action_space.sample()

        next_state, reward, done, truncated, info = env_train.step(action)

        rp.push(state, action, next_state, reward, done, truncated)

        state = next_state

        if len(rp) < batch_size:
            continue

        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch, truncateds_batch = rp.sample(batch_size)
        states_batch = states_batch.to(device)
        actions_batch = actions_batch.to(device)
        next_states_batch = next_states_batch.to(device)
        rewards_batch = rewards_batch.to(device)
        dones_batch = dones_batch.to(device)
        truncateds_batch = truncateds_batch.to(device)

        with torch.no_grad():
            future_q_values = target_network(next_states_batch)

            max_future_q = torch.max(future_q_values, dim=1)[0].detach()

            bellman_change = rewards_batch + discount_factor * max_future_q * (1 - dones_batch)

        all_current_q_values = policy_network(states_batch)

        current_q_for_actions_taken = all_current_q_values.gather(1, actions_batch.unsqueeze(1))

        loss = loss_fn(current_q_for_actions_taken, bellman_change.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_episode_losses.append(loss.item())

    if len(current_episode_losses) > 0:
        average_loss = np.mean(current_episode_losses)
        average_losses.append(average_loss)

    
plt.plot(average_losses)
plt.xlabel('Episode')
plt.ylabel('Average Loss (MSE)')
plt.title('DQN Training: Average Loss vs. Episode')
plt.grid(True)
plt.savefig("training_results.jpg")







