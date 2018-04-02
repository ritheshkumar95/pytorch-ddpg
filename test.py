import gym
from config import load_config
from modules import DDPG, to_scalar
import numpy as np
import imageio


cf = load_config('config/baseline.py')
# env = gym.make('HumanoidStandup-v2')
env = gym.make('BipedalWalker-v2')

cf.state_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.shape[0]
cf.scale = float(env.action_space.high[0])

print('Trying environment BipedalWalker-v2')
print(' State Dimensions: ', cf.state_dim)
print(' Action Dimensions: ', cf.action_dim)
print('Action low: ', env.action_space.low)
print('Action high: ', env.action_space.high)

model = DDPG(cf)
model.load_models()

frames = []
for epi in range(2):
    s_t = env.reset()
    frames.append(env.render(mode='rgb_array'))
    avg_reward = 0
    while True:
        a_t = model.sample_action(s_t)
        s_t, r_t, done, info = env.step(a_t)
        frames.append(env.render(mode='rgb_array'))
        avg_reward += r_t

        if done:
            break

print('Completed testing!')
print('No. of frames: ', len(frames))
imageio.mimwrite('trained_agent.gif', frames, fps=50)
