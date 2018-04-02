# rl parameters
max_episodes = 1000000
max_steps = 1000
max_buffer = 1000000
replay_start_size = 50000

# noise parameters
mu = 0
theta = 0.15
sigma = 0.2
dt = 1e-2

actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
batch_size = 64

gamma = 0.99
tau = 0.001
