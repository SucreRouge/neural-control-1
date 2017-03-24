import matplotlib
matplotlib.use('Agg') 

import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq import *

dt = 0.1

class System(object):
    def __init__(self):
        self.x = 0
        self.v = 0

    def update(self, control):
        self.v += -0.5*self.x * dt + control * dt
        self.x += self.v * dt

    def reset(self, test=False):
        self.x = np.random.rand()
        self.v = 0

class ControlTask(object):
    def __init__(self, system, num_actions):
        self._system      = system
        self._num_actions = num_actions
        self._setpoint    = 1.0
        self._stepcount   = 0
        self._control     = 0
        self._errors      = deque(maxlen=20)

    @property
    def state_size(self):
        # x, y, u
        return 3

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def state(self):
        return np.array([self._system.x, self._setpoint, self._control])

    def update(self, action):
        control = 2*float(action) / self.num_actions - 1
        self._system.update(control)
        self._control = control

        # calculate reward
        error = self._system.x - self._setpoint
        self._errors.append(error)
        reward = -error**2
        self._stepcount += 1
        term = False
        if self._stepcount > 300:
            term = True

        return reward, term

    def reset(self, test=False):
        self._stepcount = 0
        self._errors.clear()
        self._system.reset()
        if test:
            self._system.x = 0


################################################################################
#                             Testing Stuff
################################################################################

import matplotlib.pyplot as plt
from scipy import stats

task = ControlTask(System(), 32)

def arch(inp):
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name="fc")
    return fc

controller = DeepQController(history_length=10, memory_size=10000000, 
              state_size=task.state_size, num_actions=task.num_actions)
controller.setup_graph(arch, double_q = True)
sw = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)

def episode_callback():
    reward_hist   = deque()
    expected_hist = deque()
    def call(result):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        i = len(reward_hist)
        if i % 10 == 0:
            rwd_h = np.array(reward_hist)
            exp_h = np.array(expected_hist)
            c     = np.stack([rwd_h, exp_h]).transpose()
            np.savetxt("progress.txt", c)
    return call

def test_callback():
    reward_hist   = deque()
    expected_hist = deque()
    q_hist        = deque()
    def call(result, track):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        q_hist.append(result.mean_q)
        epoch   = controller._epoch_counter
        epsilon = controller._policy.epsilon
        acount  = controller._action_counter
        test_counter = len(reward_hist)

        # plot the test run
        fig, ax = plt.subplots(1,1)
        ax.set_title("Epoch: %d , Epsilon=%.1f%%, Score=%.2f"%(epoch, epsilon*100, result.total_reward))
        ax.set_autoscaley_on(False)
        ax.set_ylim([-1,2])
        ax.plot(track[:, 0], linewidth=2)
        ax.plot(track[:, 1])
        ax.plot([1]*len(track))
        fig.savefig("test_%d.pdf"%test_counter)
        plt.close(fig)

        rwd_h = np.array(reward_hist)
        exp_h = np.array(expected_hist)
        q_h = np.array(q_hist)
        c     = np.stack([rwd_h, exp_h, q_h]).transpose()
        np.savetxt("testing.txt", c)


    return call

run(task=task, controller=controller, num_frames=50e6, test_every=1e5, 
    episode_callback=episode_callback(), test_callback = test_callback())