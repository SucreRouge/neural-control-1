import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq import *
from deepq.copter_env import CopterEnv

import gym
dt = 0.1

################################################################################
#                             Testing Stuff
################################################################################

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

task = gym.make("Pendulum-v0")

controller = DeepPolicyGradientController(history_length=1, memory_size=1e6, 
          state_size=task.observation_space.shape[0], action_space=task.action_space,
          minibatch_size=64, final_exploration_frame=40000)

def actor(state):
    s = [d.value for d in state.get_shape()]
    flat = tf.reshape(state, [-1, s[1]*s[2]])
    flat = tf.layers.dense(flat, 400, activation=tf.nn.relu, name="fc1")
    return tf.layers.dense(flat, 300, activation=tf.nn.relu, name="fc2")

def critic(state, action):
    s = [d.value for d in state.get_shape()]
    flat = tf.reshape(state, [-1, s[1]*s[2]])
    flat = tf.layers.dense(flat, 400, activation=tf.nn.relu, name="fc1")
    state_features = tf.layers.dense(flat, 300, name="state_features")
    action_features = tf.layers.dense(action, 300, name="action_features")
    features = tf.nn.relu(tf.add(state_features, action_features, name="features"))
    return features


controller.setup_graph(actor_net = actor, critic_net = critic, target_net=True, actor_learning_rate=1e-4, 
                        critic_learning_rate=1e-3, soft_target=1e-3)


sw = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)

print(task.action_space.low)

def episode_callback(frame, result):
    print(result.total_reward)
run(task=task, controller=controller, num_frames=1e7, test_every=2e4, episode_callback=episode_callback)