import gym
from alphaatari import __gym_game_names__

for gamename in __gym_game_names__.values():
    env = gym.make(gamename)
    print(gamename)
    print(env.observation_space.shape)
    print(env.action_space.n)
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space)
# print(type(env.action_space.sample()))
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         #print(observation.shape)
#         action = env.action_space.sample()
#         #print(i_episode)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break