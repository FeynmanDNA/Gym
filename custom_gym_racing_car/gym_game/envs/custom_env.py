import gym
from gym import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D


class CustomEnv(gym.Env):
    def __init__(self):
        """
        every environment comes with an action_space
        and an observation_space

        Discrete space: fixed range of non-neg numbers
        Box space: n-dimensional box, like an array of N numbers
        * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)
        """
        self.pygame = PyGame2D()
        # two types of spaces
        # 1. Discrete space
        # speed up, turn left, turn right
        self.action_space = spaces.Discrete(3)

        # 2. Box space
        # Box is n-dimensional, like an array
        # five radar, each radar check 10 distances
        self.observation_space = spaces.Box(
            np.array([0,0,0,0,0]),
            np.array([10,10,10,10,10]),
            dtype=np.int)

    def reset(self):
        """
        delete current pygame instance,
        restart a new pygame,
        return the 1st observed data from the
        new gam
        """
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        # return the new state/observation
        return obs

    def step(self, action):
        """perform action"""
        self.pygame.action(action)
        # observation is an environment-specific
        # object representing agent observation
        # of the environment
        obs = self.pygame.observe()
        # reward is float, amount of reward by
        # previous action
        reward = self.pygame.evaluate()
        done = self.pygame.is_done() # booleadn
        # info is a dict
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """show the game"""
        self.pygame.view()
