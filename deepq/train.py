import tensorflow as tf
import numpy as np
from collections import deque
import time, math, tempfile, os, argparse
from deepq import *

import gym
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import gym_quadrotor

def ensuredir(path):
    try:
        os.makedirs(path)
    except:
        pass
    return path

parser = argparse.ArgumentParser(description='Run the copter environment')
parser.add_argument('--logdir', type=str, required=False, help='target directory for logfiles')
parser.add_argument('--max_steps', type=int, default=5e6, help='Maximum amount of steps to simulate')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Base learning rate for the algorithm')
parser.add_argument('--test_interval', type=float, default=2e4, help='How many frames between successive tests')
parser.add_argument('--render', type=bool, default=False, help='Render some episodes')
parser.add_argument('--task', type=str, default="Pendulum-v0", help='The gym environment to load')
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
    def call(result, track):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        q_hist.append(result.mean_q)
        epoch   = controller._epoch_counter
        try:
            epsilon = controller._explore_policy.epsilon
        except:
            epsilon = 0
        acount  = controller.frame_count
        test_counter = len(reward_hist)

        # plot the test run
        lp = os.path.join(testdir, "test_%d.txt--"%test_counter)
        np.savetxt(lp, track)
        rwd_h = np.array(reward_hist)
        exp_h = np.array(expected_hist)
        q_h   = np.array(q_hist)
        c     = np.stack([rwd_h, exp_h, q_h]).transpose()
        np.savetxt(tstfile, c)

        controller.save(os.path.join(logdir, "copter-demo"))

        """
        # visualize thoughts
        state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0]
        state = [state, state]
        result = np.zeros((200, 200))
        for (i, x) in enumerate(np.linspace(-1.0, 1.0, 200)):
            for (j, y) in enumerate(np.linspace(-1.0, 1.0, 200)):
                action = [i/100.0, j/100.0, 0.0, 0.0]
                q = controller.get_Q([state], [action])
                result[i, j] = q
        fig, ax = plt.subplots(1,1)
        ax.imshow(result)
        fig.savefig(os.path.join(testdir, "q_%d.pdf"%test_counter))
        plt.close(fig)
        """
    return call

def arch(inp):
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 256, activation=tf.nn.relu, name="fc")
    return fc

def PID():
    vals = [0, 0, 1, 2]
    tgts = [0, 6, 7, 8]
    controller = SimplePIDController(state_size=task.observation_space.shape[0], action_space=task.action_space,
                                     values = vals, targets = tgts)
    return controller

def DDPG(learning_rate):
    critic_regularizer = tf.contrib.layers.l2_regularizer(1e-4)
    initializer        = tf.random_uniform_initializer(-3e-3, 3e-3)
    explorative_noise  = noise.OrnsteinUhlenbeckNoise(mu = 0.0, theta = 0.15, sigma=0.2)

    controller = DeepPolicyGradientController(history_length=2, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=task.action_space,
              minibatch_size=64, final_exploration_frame=2e6, final_epsilon=0.05,
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
    critic_lr = tf.train.exponential_decay(learning_rate, gstep, 100000, 0.95, staircase=True)
    policy_lr = tf.train.exponential_decay(learning_rate*1e-1, gstep, 100000, 0.95, staircase=True)

    controller.setup_graph(actor_net = actor, critic_net = critic, actor_learning_rate=policy_lr, 
                            critic_learning_rate=critic_lr, soft_target=1e-3, global_step=gstep,
                            critic_regularizer=critic_regularizer, critic_init=initializer,
                            policy_init=initializer)
    return controller

def DQN(learning_rate):
    controller = DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=ActionSpace(task.action_space).discretized(3),
              final_exploration_frame=2e5, minibatch_size=32)
    controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=learning_rate)
    return controller

def MultiDQN(learning_rate):
    controllers = [DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=action_space.spaces.Discrete(9),
              steps_per_epoch=20000, final_exploration_frame=5e5, minibatch_size=32) 
                  for i in range(4)]
    for (i, controller) in enumerate(controllers):
        with tf.variable_scope("controller_%d"%i):
            controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=learning_rate)
    controller = NaiveMultiController(controllers, ActionSpace(task.action_space).discretized(9))
    return controller


task = gym.make(args.task)
env = gym.wrappers.Monitor(task, directory=os.path.join(logdir, "monitor"), force=True)
controller = DDPG(learning_rate = args.learning_rate)    

if args.render:
    def cb(task): task.render(mode='human')
else:
    def cb(task): pass
sw = tf.summary.FileWriter(logdir, graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)
run(task=env, controller=controller, num_frames=args.max_steps, test_every=args.test_interval, 
    episode_callback=episode_callback(), test_callback = test_callback(), logdir=logdir,
    test_step_callback=cb, train_step_callback=cb)
