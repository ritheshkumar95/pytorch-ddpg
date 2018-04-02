import gym
from config import load_config
from modules import DDPG, OrnsteinUhlenbeckNoise, to_scalar
import numpy as np
from itertools import count


cf = load_config('config/baseline.py')
# env = gym.make('HalfCheetah-v2')
env = gym.make('BipedalWalker-v2')

cf.state_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.shape[0]
cf.scale = float(env.action_space.high[0])

print('Trying environment BipedalWalker-v2')
print(' State Dimensions: ', cf.state_dim)
print(' Action Dimensions: ', cf.action_dim)
print('Action low: ', env.action_space.low)
print('Action high: ', env.action_space.high)

noise_process = OrnsteinUhlenbeckNoise(cf)
model = DDPG(cf)
model.copy_weights(model.actor, model.actor_target)
model.copy_weights(model.critic, model.critic_target)

losses = []
total_timesteps = 0
for epi in range(cf.max_episodes):
    s_t = env.reset()
    noise_process.reset()
    avg_reward = 0
    for t in count(1):
        a_t = model.sample_action(s_t)
        a_t = a_t + noise_process.sample()

        s_tp1, r_t, done, info = env.step(a_t)

        model.buffer.add(s_t, a_t, r_t, s_tp1, float(done == False))
        avg_reward += r_t

        if done:
            break
        else:
            s_t = s_tp1

        if model.buffer.len >= cf.replay_start_size:
            _loss_a, _loss_c = model.train_batch()
            losses.append(to_scalar([_loss_a, _loss_c]))

    if len(losses) > 0:
        total_timesteps += t
        avg_loss_a, avg_loss_c = np.asarray(losses)[-100:].mean(0)
        print(
            'Episode {}: actor loss: {} critic loss: {}\
            total_reward: {} timesteps: {} tot_timesteps: {}'.format(
             epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps
            ))

    if (epi + 1) % 200 == 0:
        model.save_models()
print('Completed training!')
