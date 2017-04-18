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
import math
from collections import deque

logger = logging.getLogger(__name__)

class CopterParams(object):
    def __init__(self):
        self.l = 0.31    # Arm length
        self.b = 5.324e-5    # Thrust coefficient
        self.d = 8.721e-7    # Drag coefficient
        self.m = 0.723    # Mass
        self.I = np.array([[8.678e-3,0,0],[0,8.678e-3,0],[0,0,3.217e-2]]) # Inertia
        self.J = 7.321e-5   # Rotor inertia


class CopterStatus(object):
    def __init__(self):
        self.position = np.array([0.0, 0, 0])
        self.velocity = np.array([0.0, 0, 0])
        self.attitude = np.array([0.0, 0, 0])
        self.angular_velocity = np.array([0.0, 0, 0])


class CopterEnv(gym.Env):
    metadata = {
        'render.modes': [],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        high = np.array([np.inf]*15)
        
        self.copterparams = CopterParams()
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(-1, 1, (4,))

        self.target         = np.zeros(3)
        self.threshold      =  2 * math.pi / 180
        self.fail_threshold = 15 * math.pi / 180

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        control = np.array(action) * 0.05
        dt      = 0.1

        ap, aa  = self._calc_acceleration(control)
        quad    = self.copterstatus
        quad.position += quad.velocity * dt + 0.5 * ap * dt * dt
        quad.velocity += ap * dt

        quad.attitude += quad.angular_velocity * dt + 0.5 * aa * dt * dt
        quad.angular_velocity += aa * dt

        err = np.max(np.abs(quad.attitude - self.target))

        self._steps += 1
        done = bool(self._steps > 1000)

        # positive reward for not falling over
        reward = 0.2 * (1 - err / self.fail_threshold)
        if err < self.threshold:
            merr = np.mean(np.abs(quad.attitude - self.target)) # this is guaranteed to be smaller than err
            rerr = merr / self.threshold
            reward += 1.1 - rerr

        # reward for keeping velocities low
        velmag = np.mean(np.abs(quad.angular_velocity))
        reward += max(0.0, 0.1 - velmag)

        # reward for constant control
        cchange = np.mean(np.abs(control - self._last_control))
        reward += max(0, 0.1 - 2*cchange)

        # normalize reward so that we can get at most 1.0 per step
        reward /= 1.5

        if err > self.fail_threshold or quad.position[2] < 0.0 or quad.position[2] > 10:
            reward = -10
            done = True

        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.angular_velocity += self.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

        if self.np_random.rand() < 0.01:
            self.target += self.np_random.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

        self._last_control = control
        return self._get_state(), reward, done, {}

    def _calc_acceleration(self, control):
        b = self.copterparams.b
        I = self.copterparams.I
        l = self.copterparams.l
        m = self.copterparams.m
        J = self.copterparams.J
        d = self.copterparams.d
        g = 9.81

        attitude = self.copterstatus.attitude
        avel     = self.copterstatus.angular_velocity
        roll     = attitude[0]
        pitch    = attitude[1]
        yaw      = attitude[2]

        droll    = avel[0]
        dpitch   = avel[1]
        dyaw     = avel[2]

        # damn, have to calculate this
        U1s = control[0] / b
        U2s = control[1] / b
        U3s = control[2] / b
        U4s = control[3] / d
        U13 = (U1s + U4s) / 2
        U24 = (U1s - U4s) / 2
        O1 = math.sqrt(abs(U13 + U3s)/2)
        O3 = math.sqrt(abs(U13 - U3s)/2)
        O2 = math.sqrt(abs(U24 - U2s)/2)
        O4 = math.sqrt(abs(U24 + U2s)/2)
        Or = -O1 + O2 - O3 + O4

        c0 =  (4*control[0] + 1.0) *  m*g
        a0  = c0 * ( math.cos(roll)*math.sin(pitch)*math.cos(yaw) + math.sin(roll)*math.sin(yaw) ) / m
        a1  = c0 * ( math.cos(roll)*math.sin(pitch)*math.sin(yaw) + math.sin(roll)*math.cos(yaw) ) / m
        a2  = c0 * ( math.cos(roll)*math.cos(pitch) ) / m - g

        
        aroll  = (dpitch * dyaw * (I[1, 1] - I[2, 2]) + dpitch * Or * J + control[1] * l) / I[0, 0]
        apitch = (droll  * dyaw * (I[2, 2] - I[0, 0]) + droll * Or * J  + control[2] * l) / I[1, 1]
        ayaw   = (droll  * dyaw * (I[0, 0] - I[1, 1]) + control[3] * l) / I[2, 2]
        return np.array([a0, a1, a2]), np.array([aroll, apitch, ayaw])

    def _reset(self):
        self.copterstatus = CopterStatus()
        # start in resting position, but with low angular velocity
        self.copterstatus.angular_velocity = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.velocity         = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.position         = np.array([0.0, 0, 1])
        self.target = self.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180
        self.copterstatus.attitude = self.target + self.np_random.uniform(low=-5, high=5, size=(3,)) * math.pi / 180
        self._steps = 0
        self._last_control = np.zeros(4)

        return self._get_state()

    def _get_state(self):
        s = self.copterstatus
        # currently, we ignore position and velocity!
        return np.concatenate([s.attitude, s.angular_velocity, self.target, s.position, s.velocity])

    def _render(self, mode='human', close=False):
        # currently not implemented
        return
