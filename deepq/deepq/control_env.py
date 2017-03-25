"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class ControlEnv(gym.Env):
    metadata = {
        'render.modes': [],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        high = np.array([np.inf, np.inf, 1])
        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(-high, high)

        self.threshold = 0.1

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        control = (action  - 8) / 8.0
        dt      = 0.1

        # integrate state
        self._v += -0.5*self._x * dt + control * dt
        self._x += self._v * dt
        self._c = control
        self._steps += 1

        err = abs(self._x - self._y)

        done = bool(self._steps > 300)

        if err < self.threshold:
            reward = 1.0
        else:
            reward = -1

        return self._get_state(), reward, done, {}

    def _reset(self):
        self._x = self.np_random.uniform(low=0.0, high=1.0, size=(1,))[0]
        self._v = self.np_random.uniform(low=-0.05, high=0.05, size=(1,))[0]
        self._c = 0.0
        self._y = 1.0
        self._steps = 0

        self.steps_beyond_done = None
        return self._get_state()

    def _get_state(self):
        return np.array([self._x, self._y, self._c])

    def _render(self, mode='human', close=False):
        # currently not implemented
        return