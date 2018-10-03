import gym
from alphaatari import __gym_game_names__
import skimage as skimage
from skimage import transform, color, exposure

import matplotlib.pyplot as plt

for gamename in __gym_game_names__.values():
    env = gym.make(gamename)
    print(gamename)
    print(env.observation_space.shape)
    print(env.action_space.n)

observation_color = env.reset()
x = skimage.color.rgb2gray(observation_color)
#x = skimage.transform.resize(x,(160,320))
x = skimage.exposure.rescale_intensity(x, out_range=(0,1))
print(x.shape)

plt.figure()
plt.imshow(x)
plt.show()
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