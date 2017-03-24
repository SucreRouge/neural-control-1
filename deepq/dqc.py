import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq.memory import *
from deepq.qnet import QNet

dt = 0.1

class System(object):
    def __init__(self):
        self.x = 0
        self.v = 0

    def update(self, control):
        self.v += -0.5*self.x * dt + control * dt
        self.x += self.v * dt

    def reset(self):
        self.x = 0
        self.v = 0


class EGreedy(object):
    def __init__(self, eps):
        self.epsilon = eps

    def __call__(self, actions):
        if np.random.rand() < self.epsilon:
            a = np.random.randint(len(actions[0]))
            return a
        else:
            return np.argmax(actions)


class QLearner(object):
    def __init__(self, history_length, memory_size, state_size, num_actions, session):
        self._num_actions    = num_actions
        self._state_size     = state_size
        self._history_length = history_length
        self._history        = History(duration=history_length, state_size=state_size)
        self._state_memory   = Memory(size=memory_size, history_length=history_length, state_size=state_size)
        self._session        = session
        self._policy         = EGreedy(1.0)
        self._step_counter   = 0
        self._epoch_counter  = 0
        self._steps_per_epoch = 10000
        self._last_action    = None
        self._next_epoch     = None

    def observe(self, state, reward):
        # if this is the first state, there is no transition to remember,
        # so simply add to the state history
        if self._last_action is None:
            self._history.observe(state)
            return

        terminal = state is None
        last_state = self._history.state
        action = self._last_action
        if not terminal:
            next_state = self._history.observe(state)
        else:
            next_state = None
            self._last_action = None
            self._history.clear()
        self._state_memory.append(state=last_state, next=next_state, reward=reward, action=action)

    def get_action(self):
        full_state = self._history.state
        action_vals   = self._qnet.get_actions(full_state, self._session)
        action        = self._policy(action_vals)
        self._last_action = action
        
        return action, action_vals

    def train(self):
        if len(self._state_memory) < 1000:
            return

        ls = self._qnet.train_step(self._state_memory.sample(32), self._session)
        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            # copy target net to policy net
            self._qnet.update_target(self._session)
            self._step_counter = 0
            self._epoch_counter += 1
            if self._next_epoch:
                self._next_epoch()

    def setup_graph(self, arch):
        qnet = QNet(state_size     = self._state_size, 
                    history_length = self._history_length, 
                    num_actions    = self._num_actions)

        # TODO Figure these out!
        opt = tf.train.AdamOptimizer()
        self._qnet = qnet.build_graph(arch, opt)

    def init(self):
        self._session.run([tf.global_variables_initializer()])
        self._qnet.update_target(self._session)

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

    def reset(self):
        self._stepcount = 0
        self._errors.clear()
        self._system.reset()


################################################################################
#                             Testing Stuff
################################################################################

import matplotlib.pyplot as plt
from scipy import stats

task = ControlTask(System(), 32)

def arch(inp):
    c1 = tf.layers.conv1d(inp, 32, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc1 = tf.layers.dense(flat, 50, activation=tf.nn.relu, name="fc1")
    fc2 = tf.layers.dense(fc1,  30, activation=tf.nn.relu, name="fc2")
    return fc2

ql = QLearner(history_length=10, memory_size=50000, 
              state_size=task.state_size, num_actions=task.num_actions, 
              session=tf.Session())

def observer():
    epoch_rewards = deque()
    reward_history = deque()
    q_history = deque()
    epoch_q = deque()
    def update_epoch():
        reward_history.append(np.mean(epoch_rewards))
        q_history.append(np.mean(epoch_q))
        epoch_rewards.clear()
        epoch_q.clear()
    return reward_history, update_epoch, epoch_rewards, q_history, epoch_q

ql.setup_graph(arch)
ql.init()
ct = deque()
fig, ax = plt.subplots(2,1)
# observe the initial state
ql.observe(task.state, 0)

reward_history, ue, epoch_rewards, q_history, epoch_q = observer()
ql._next_epoch = ue

for i in range(10000000):
    action, values = ql.get_action()
    reward, terminal = task.update(action)
    epoch_rewards.append(reward)
    epoch_q.append(np.amax(values))
    ql.observe(None if terminal else task.state, reward)
    if terminal:
        task.reset()

    ql.train()
    if task._stepcount == 0:
        ax[0].clear()
        ax[0].set_title("Epoch: %d , Epsilon=%f"%(ql._epoch_counter, ql._policy.epsilon))
        ax[0].plot(np.array(ct)[:, 0], linewidth=2)
        ax[0].plot(np.array(ct)[:, 1])
        ax[0].plot([1]*len(ct))
        plt.pause(0.0001)

        print(reward_history)
        if len(reward_history) > 1:
            ax[1].clear()
            h = np.array(reward_history)
            hist = min(len(h), 25)
            x = np.arange(len(h)-hist, len(h))
            slope, intercept, _, _, _ = stats.linregress(x, h[-hist:])
            line = slope*x+intercept
            ax[1].plot(h)
            ax[1].plot(q_history)
            ax[1].plot(x, line)
        
        ct.clear()
    ct.append(np.array([task._system.x, task._control]))

# don't exit yet
plt.show()