from collections import deque
import torch
import random

class ReplayBuffer:
    def __init__ (self, capacity):
        self.dq = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done, truncated):
        self.dq.append((state, action, next_state, reward, done, truncated))

    def sample(self, batch_size):
        memories = random.sample(self.dq, batch_size)

        states, actions, next_states, rewards, dones, truncateds = zip(*memories)

        # print(states)
        # print(actions)
        # print(next_states)
        # print(rewards)
        # print(dones)
        # print(truncateds)
        # print(torch.tensor(states).reshape(batch_size, 4))

        states = torch.tensor(states).reshape(batch_size, 4)
        actions = torch.tensor(actions).reshape(batch_size, 1)
        next_states = torch.tensor(next_states).reshape(batch_size, 4)
        rewards = torch.tensor(rewards).reshape(batch_size, 1)
        dones = torch.tensor(dones).reshape(batch_size, 1)
        truncateds = torch.tensor(truncateds).reshape(batch_size, 1)

        return states, actions, next_states, rewards, dones, truncateds
    
    def __len__(self):
        return len(self.dq)
    

