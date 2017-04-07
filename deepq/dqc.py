import matplotlib
matplotlib.use('Agg') 

import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq import *
from deepq.control_env import ControlEnv

import gym
dt = 0.1

################################################################################
#                             Testing Stuff
################################################################################

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

task = ControlEnv()

def arch(inp):
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 128, activation=tf.nn.relu, name="fc")
    return fc

def episode_callback():
    reward_hist   = deque()
    expected_hist = deque()
    def call(result):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        i = len(reward_hist)
        if i % 100 == 0:
            rwd_h = np.array(reward_hist)
            exp_h = np.array(expected_hist)
            c     = np.stack([rwd_h, exp_h]).transpose()
            np.savetxt("progress.txt", c)
            if len(reward_hist) > 100:
                print(np.mean(np.array(reward_hist)[-100:]))
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
        acount  = controller.frame_count
        test_counter = len(reward_hist)

        # plot the test run
        fig, ax = plt.subplots(1,1)
        ax.set_title("Epoch: %d , Epsilon=%.1f%%, Score=%.2f"%(epoch, epsilon*100, result.total_reward))
        ax.set_autoscaley_on(False)
        ax.set_ylim([-2, 2])
        ax.plot(track[:, 4])               # y
        ax.plot(track[:, 5])               # c
        ax.plot(track[:, 3])               # x
        fig.savefig("test_%d.pdf"%test_counter)
        plt.close(fig)

        rwd_h = np.array(reward_hist)
        exp_h = np.array(expected_hist)
        q_h = np.array(q_hist)
        c     = np.stack([rwd_h, exp_h, q_h]).transpose()
        np.savetxt("testing.txt", c)


    return call

controller = DiscreteDeepQController(history_length=10, memory_size=1000000, 
              state_size=task.observation_space.shape[0], action_space=task.action_space,
              final_exploration_frame=1e5, minibatch_size=32)
controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=2.5e-4)
sw = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)

run(task=task, controller=controller, num_frames=1e6, test_every=1e4, 
    episode_callback=episode_callback(), test_callback = test_callback())