import gymnasium as gym
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn

env_train = gym.make("CartPole-v1")
env_test = gym.make("CartPole-v1", render_mode="human")



