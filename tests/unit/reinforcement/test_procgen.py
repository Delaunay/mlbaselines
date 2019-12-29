
import numpy as np
import gym
from procgen import ProcgenEnv

# env = gym.make('procgen:procgen-coinrun-v0')
# obs = env.reset()
#
# while True:
#     obs, rew, done, info = env.step(env.action_space.sample())
#     env.render()
#     if done:
#         break


env = ProcgenEnv(num_envs=2, env_name="coinrun", num_levels=12, start_level=34)
obs = env.reset()

print(obs['rgb'].shape)


action = np.ones(2) * env.action_space.sample()

obs, rew, done, info = env.step(action)

print(obs)
print(rew)
print(done)
print(info)

# while True:
#     obs, rew, done, info = env.step(env.action_space.sample())
#     env.render()
#     if done:
#         break
