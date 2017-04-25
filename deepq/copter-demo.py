import tensorflow as tf
import numpy as np
from collections import deque
import time, math, tempfile, os, argparse
from deepq import *
from deepq.copter_env import CopterEnv

import gym
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt

def ensuredir(path):
    try:
        os.makedirs(path)
    except:
        pass
    return path

parser = argparse.ArgumentParser(description='Run the copter environment')
parser.add_argument('-logdir', type=str, required=False, help='target directory for logfiles')
args = parser.parse_args()

if args.logdir is None:
    logdir = tempfile.mkdtemp(dir="./logs/")
else:
    logdir = ensuredir(args.logdir)

testdir = ensuredir(os.path.join(logdir, "tests"))
prgfile = os.path.join(logdir, "progress.txt")
tstfile = os.path.join(logdir, "testing.txt")

def episode_callback():
    reward_hist   = deque()
    expected_hist = deque()
    frame_hist    = deque()
    def call(frame, result):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        frame_hist.append(frame)
        i = len(reward_hist)
        if i % 10 == 0:
            rwd_h = np.array(reward_hist)
            exp_h = np.array(expected_hist)
            frm_h = np.array(frame_hist)
            c     = np.stack([frm_h, rwd_h, exp_h]).transpose()
            np.savetxt(prgfile, c)
            if len(reward_hist) > 100:
                print(np.mean(np.array(reward_hist)[-100:]))
    return call

def test_callback():
    reward_hist   = deque()
    expected_hist = deque()
    q_hist        = deque()
    fails         = deque()
    def call(result, track):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        q_hist.append(result.mean_q)
        fails.append(task._fail_count)
        epoch   = controller._epoch_counter
        try:
            epsilon = controller._policy.epsilon
        except:
            epsilon = 0
        acount  = controller.frame_count
        test_counter = len(reward_hist)

        # plot the test run
        fig, ax = plt.subplots(1,1)
        ax.set_title("Epoch: %d , Epsilon=%.1f%%, Score=%.2f"%(epoch, epsilon*100, result.total_reward))
        ax.set_autoscaley_on(False)
        ax.set_ylim([-25, 25])

        ax.plot(track[:, 6] * 180/math.pi)              # y
        ax.plot(track[:, 7] * 180/math.pi)              # c
        ax.plot(track[:, 8] * 180/math.pi)              # c
        ax.plot(track[:, 12] * 180/math.pi)              # x
        ax.plot(track[:, 13] * 180/math.pi)              # x
        ax.plot(track[:, 14] * 180/math.pi)              # x
        fig.savefig(os.path.join(testdir, "test_%d.pdf"%test_counter))
        plt.close(fig)

        rwd_h = np.array(reward_hist)
        exp_h = np.array(expected_hist)
        q_h   = np.array(q_hist)
        fls   = np.array(fails)
        c     = np.stack([rwd_h, exp_h, q_h, fls]).transpose()
        np.savetxt(tstfile, c)

        controller.save(os.path.join(logdir, "copter-demo"))


    return call

def arch(inp):
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 256, activation=tf.nn.relu, name="fc")
    return fc

task = CopterEnv()

def PID():
    vals = [0, 0, 1, 2]
    tgts = [0, 6, 7, 8]
    controller = SimplePIDController(state_size=task.observation_space.shape[0], action_space=task.action_space,
                                     values = vals, targets = tgts)
    return controller

def DDPG():
    critic_regularizer = tf.contrib.layers.l2_regularizer(1e-4)
    initializer        = tf.random_uniform_initializer(-3e-3, 3e-3)
    explorative_noise  = noise.OrnsteinUhlenbeckNoise(mu = 0.0, theta = 0.15, sigma=0.2)

    controller = DeepPolicyGradientController(history_length=2, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=task.action_space,
              minibatch_size=64, final_exploration_frame=1e6, final_epsilon=0.05,
              explorative_noise=explorative_noise)

    def actor(state):
        s = [d.value for d in state.get_shape()]
        flat = tf.reshape(state, [-1, s[1]*s[2]], name="state")
        reg = tf.contrib.layers.l2_regularizer(1e-4)
        flat = tf.layers.dense(flat, 400, activation=tf.nn.relu, kernel_regularizer=reg, name="fc1")
        return tf.layers.dense(flat, 300, activation=tf.nn.relu, kernel_regularizer=reg, name="fc2")

    def critic(state, action):
        s = [d.value for d in state.get_shape()]
        flat = tf.reshape(state, [-1, s[1]*s[2]], name="state")
        reg = critic_regularizer
        params = {'activation':  tf.nn.relu,
                  'kernel_regularizer': reg}

        flat = tf.layers.dense(flat, 400, name="fc1", **params)
        state_features  = tf.layers.dense(flat,   300, name="state_features",  **params)
        action_features = tf.layers.dense(action, 300, name="action_features", **params)
        features = tf.add(state_features, action_features, name="features")

        features = tf.layers.dense(features, 300, name="features2", **params)
        return features

    # decaing learning rates
    gstep     = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    critic_lr = tf.train.exponential_decay(1e-3, gstep, 100000, 0.95, staircase=True)
    policy_lr = tf.train.exponential_decay(1e-4, gstep, 100000, 0.95, staircase=True)

    controller.setup_graph(actor_net = actor, critic_net = critic, actor_learning_rate=policy_lr, 
                            critic_learning_rate=critic_lr, soft_target=1e-3, global_step=gstep,
                            critic_regularizer=critic_regularizer, critic_init=initializer,
                            policy_init=initializer)
    return controller

def DQN():
    controller = DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=ActionSpace(task.action_space).discretized(3),
              final_exploration_frame=2e5, minibatch_size=32)
    controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=2.5e-4)
    return controller

def MultiDQN():
    controllers = [DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=action_space.spaces.Discrete(9),
              steps_per_epoch=20000, final_exploration_frame=5e5, minibatch_size=32) 
                  for i in range(4)]
    for (i, controller) in enumerate(controllers):
        with tf.variable_scope("controller_%d"%i):
            controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=1.0e-4)
    controller = NaiveMultiController(controllers, ActionSpace(task.action_space).discretized(9))
    return controller



controller = DDPG()    

sw = tf.summary.FileWriter(logdir, graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)
run(task=task, controller=controller, num_frames=2e6, test_every=2e4, 
    episode_callback=episode_callback(), test_callback = test_callback())
