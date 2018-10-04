import gym
import time
import numpy as np

import skimage as skimage
from skimage import transform, color, exposure

class AtariGame:
    '''
    A common class for Atari game
    '''
    def __init__(self, game_name, observation_shape=(100,75), verbose=False):
        self.game_name = game_name
        self.verbose = verbose
        self.observation_shape = observation_shape

        self.observation = None
        self.env = gym.make(self.game_name)
        
        self.reset()

    def get_observation_shape(self):
        '''
        Get the shape of observation
        '''
        #return np.copy(self.env.observation_space.shape[:2])
        return np.copy(self.observation_shape)

    def get_action_dimension(self):
        '''
        Get the dimension of action
        '''
        return self.env.action_space.n

    def render(self):
        '''
        Render game in screen
        '''
        return self.env.render()

    def set_event(self, key_press, key_release):
        self.env.unwrapped.viewer.window.on_key_press = key_press
        self.env.unwrapped.viewer.window.on_key_release = key_release

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
        x = skimage.transform.resize(x, self.observation_shape, mode='reflect',anti_aliasing=True)
        #print(x.shape)
        #x = skimage.exposure.rescale_intensity(x, out_range=(0,255))
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

class GameEngine:
    def __init__(self, game_name, verbose=False):
        self.game_name = game_name
        self.verbose = verbose

        self.atarigame = AtariGame(game_name=self.game_name, verbose=verbose)
        self.human_agent_action = 0
        self.ACTIONS = self.atarigame.get_action_dimension()

    def key_press(self, key, mod):
        if int(key) == 65363:
            action = 2
        if int(key) == 65361:
            action = 3
        if int(key) == 32:
            action = 1

        if int(key) not in [32, 65361, 65363]: return
        self.human_agent_action = action

    def key_release(self, key, mod):
        if int(key) == 65363:
            action = 2
        if int(key) == 65361:
            action = 3
        if int(key) == 32:
            action = 1

        if int(key) not in [32, 65361, 65363]: return
        if self.human_agent_action == action:
            self.human_agent_action = 0

    def start(self):
        self.atarigame.render()
        self.atarigame.set_event(key_press=self.key_press, key_release=self.key_release)
        self.atarigame.reset()

        total_timesteps = 0
        total_reward = 0

        if self.verbose:
            print("Start to play Atari game and render it on the screen.")

        while 1:
            total_timesteps += 1
            obser, r, done, info = self.atarigame.step(self.human_agent_action)
            if r != 0:
                print("Action [{0}] with reward [{1:0.3f}]".format(self.human_agent_action, r))
            total_reward += r 

            window_still_open = self.atarigame.render()
            if window_still_open==False: return False
            if done: break
            time.sleep(0.1)

        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

    def start_ai(self, ai):
        if self.verbose:
            print("Watch AI to play Atari game and render it on the screen.")

        # TODO the process for AI model to play Atari game

        pass
