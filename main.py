import numpy as np
import gym
import torch

# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()

device = torch.device('cpu')
x = torch.linspace(-np.pi, np.pi, 2000, device=device, dtype=torch.float)
y = torch.sin(x)


















































































































































































































































































































