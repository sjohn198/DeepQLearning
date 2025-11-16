import gymnasium as gym
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

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
epsilon_decay_rate = 1 / total_episodes
min_epsilon = 0.001

learning_rate = 0.001
discount_factor = 0.99

batch_size = 64
rp = ReplayBuffer(10000)

policy_network = QPolicyNetwork().to(device)
target_network = QPolicyNetwork().to(device)
target_network.load_state_dict(policy_network.state_dict())

target_update_frequency = 30

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
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_network(state_tensor)
            action = torch.argmax(q_values).item()
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
            # print(max_future_q.shape)
            # print("Rewards", rewards_batch.shape)
            # print("Dones", dones_batch.shape)

            bellman_change = rewards_batch + discount_factor * max_future_q * (1 - dones_batch)

            # print(bellman_change.shape)

        all_current_q_values = policy_network(states_batch)

        current_q_for_actions_taken = all_current_q_values.gather(1, actions_batch)

        loss = loss_fn(current_q_for_actions_taken, bellman_change.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_episode_losses.append(loss.item())

    if len(current_episode_losses) > 0:
        average_loss = np.mean(current_episode_losses)
        average_losses.append(average_loss)

    if (episode + 1) % target_update_frequency == 0:
        target_network.load_state_dict(policy_network.state_dict())

    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)


    print(episode)

    
plt.plot(average_losses)
plt.xlabel('Episode')
plt.ylabel('Average Loss (MSE)')
plt.title('DQN Training: Average Loss vs. Episode')
plt.grid(True)
plt.savefig("training_results.jpg")


for i in range(5):
    state, info = env_test.reset()
    done = False
    truncated = False

    total_reward = 0

    while not (done or truncated):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = policy_network(state_tensor)

        action = torch.argmax(q_values).item()
        next_state, reward, done, truncated, info = env_test.step(action)

        state = next_state
        total_reward += reward

        print(i)

        time.sleep(0.02)
    

env_train.close()
env_test.close()

