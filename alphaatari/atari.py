import gym
import numpy as np

import skimage as skimage
from skimage import transform, color, exposure

class AtariGame:
    '''
    A common class for Atari game
    '''
    def __init__(self, game_name, verbose=False):
        self.game_name = game_name
        self.verbose = verbose

        self.observation = None
        self.env = gym.make(self.game_name)
        
        self.reset()

    def get_observation_shape(self):
        '''
        Get the shape of observation
        '''
        return np.copy(self.env.observation_space.shape[:2])

    def get_action_dimension(self):
        '''
        Get the dimension of action
        '''
        return self.env.action_space.n

    def render(self):
        '''
        Render game in screen
        '''
        self.env.render()

    def reset(self):
        '''
        Reset game
        '''
        self.observation = self.env.reset()

    def get_observation(self):
        '''
        Get observation in gray
        '''
        x = skimage.color.rgb2gray(self.observation)
        x = skimage.exposure.rescale_intensity(x, out_range=(0, 1))
        return x

    def get_random_action(self):
        '''
        Get random action
        '''
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        '''
        Make action and return the status of game
        '''
        observation, reward, done, info = self.env.step(action)
        self.observation = observation
        return observation, reward, done, info