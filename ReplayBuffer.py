from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__ (self, capacity):
        self.dq = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done, truncated):
        self.dq.append((state, action, next_state, reward, done, truncated))

    def sample(self, batch_size):
        memories = random.sample(self.dq, batch_size)

        states, actions, next_states, rewards, dones, truncated = zip(*memories)

        states = np.array(states).reshape(batch_size, 4)
        actions = np.array(actions).reshape(batch_size, 1)
        next_states = np.array(next_states).reshape(batch_size, 4)
        rewards = np.array(rewards).reshape(batch_size, 1)
        dones = np.array(dones).reshape(batch_size, 1)
        truncateds = np.array(truncateds).reshape(batch_size, 1)

        return states, actions, next_states, rewards, dones, truncateds
    
    def __len__(self):
        return len(self.dq)
    

