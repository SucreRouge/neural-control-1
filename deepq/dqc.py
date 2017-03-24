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

    def reset(self, test=False):
        self.x = np.random.rand()
        self.v = 0


class EGreedy(object):
    def __init__(self, start_eps, end_eps, num_steps):
        self._start_epsilon = start_eps
        self._end_epsilon   = end_eps
        self._num_steps     = num_steps
        self.epsilon = start_eps

    def set_stepcount(self, steps):
        decay = min(1, float(steps) / self._num_steps)
        dist  = self._start_epsilon - self._end_epsilon
        self.epsilon = self._start_epsilon - decay * dist

    def __call__(self, actions, test=False):
        if not test and np.random.rand() < self.epsilon:
            a = np.random.randint(len(actions[0]))
            return a
        else:
            return np.argmax(actions)


class QLearner(object):
    def __init__(self, history_length, memory_size, state_size, num_actions, session,
                    steps_per_epoch=10000, final_exploration_frame=1000000):
        # configuration variables (these remain constant)
        self._num_actions    = num_actions
        self._state_size     = state_size
        self._history_length = history_length
        self._steps_per_epoch = steps_per_epoch
        self._next_epoch     = None
        self._policy         = EGreedy(1.0, 0.1, final_exploration_frame)
        
        self._history        = History(duration=history_length, state_size=state_size)
        self._state_memory   = Memory(size=memory_size, history_length=history_length, state_size=state_size)
        self._session        = session
        self._last_action    = None

        # counters
        self._action_counter = 0
        self._step_counter   = 0
        self._epoch_counter  = 0

    def observe(self, state, reward, test=False):
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
        
        # if not in test mode, remember the transition
        if not test:
            self._state_memory.append(state=last_state, next=next_state, reward=reward, action=action)

    def get_action(self, test=False):
        full_state    = self._history.state
        action_vals   = self._qnet.get_actions(full_state, self._session)
        action        = self._policy(action_vals, test)
        self._last_action = action
        if not test:
            self._action_counter += 1
            self._policy.set_stepcount(self._action_counter)
        
        return action, action_vals

    def train(self, summary_writer=None):
        if len(self._state_memory) < 10000:
            return

        sample = self._state_memory.sample(32)
        ls = self._qnet.train_step(sample, self._session, summary_writer)
        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            # copy target net to policy net
            self._qnet.update_target(self._session)
            self._step_counter = 0
            self._epoch_counter += 1
            if self._next_epoch:
                self._next_epoch()

    def setup_graph(self, arch, double_q=False):
        qnet = QNet(state_size     = self._state_size, 
                    history_length = self._history_length, 
                    num_actions    = self._num_actions,
                    double_q       = double_q)

        # TODO Figure these out!
        opt = tf.train.RMSPropOptimizer(learning_rate=1.0e-4, decay=0.99, epsilon=0.01, momentum=0.95)
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
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc1 = tf.layers.dense(flat, 100, activation=tf.nn.relu, name="fc1")
    fc2 = tf.layers.dense(fc1,  50, activation=tf.nn.relu, name="fc2")
    return fc2

ql = QLearner(history_length=10, memory_size=10000000, 
              state_size=task.state_size, num_actions=task.num_actions, 
              session=tf.Session())

next_ep = False

def observer():
    reward_history = deque()
    q_history = deque()
    def update_epoch():
        episode_rewards = deque()
        episode_q = deque()
        ct = deque()
        # start a new task here
        task.reset()
        task._system.x = 0 # make the task deterministic
        total_reward = 0
        while True:
            action, values = ql.get_action(test=True)
            reward, terminal = task.update(action)
            total_reward += reward
            episode_rewards.append(reward)
            episode_q.append(total_reward + np.amax(values))
            ql.observe(None if terminal else task.state, reward, test=True)
            if terminal:
                task.reset()
                break
            ct.append(np.array([task._system.x, task._control]))
        reward_history.append(np.sum(episode_rewards))
        q_history.append(np.mean(episode_q))
        fig, ax = plt.subplots(1,1)
        ax.set_title("Epoch: %d , Epsilon=%.1f%%, Score=%.2f"%(ql._epoch_counter, ql._policy.epsilon*100, np.sum(episode_rewards)))
        ax.set_autoscaley_on(False)
        ax.set_ylim([-1,2])
        ax.plot(np.array(ct)[:, 0], linewidth=2)
        ax.plot(np.array(ct)[:, 1])
        ax.plot([1]*len(ct))
        fig.savefig("test_%d.pdf"%ql._epoch_counter)
        plt.close(fig)
        global next_ep
        next_ep = True

    return reward_history, update_epoch, q_history

ql.setup_graph(arch, double_q = True)
ql.init()
sw = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph(), flush_secs=30)
ct = deque()
fig, ax = plt.subplots(2,1)
# observe the initial state
ql.observe(task.state, 0)

reward_history, ue, q_history = observer()
ql._next_epoch = ue

for i in range(10000000):
    action, values = ql.get_action()
    reward, terminal = task.update(action)
    ql.observe(None if terminal else task.state, reward)
    if terminal:
        task.reset()

    if i % 4 == 0:
        ql.train(sw if i % 100 == 0 else None)
    if task._stepcount == 0:
        plt.pause(0.0001)
        ax[0].clear()
        ax[0].set_autoscaley_on(False)
        ax[0].set_ylim([-1,2])
        ax[0].set_title("Epoch: %d , Epsilon=%.1f%%"%(ql._epoch_counter, ql._policy.epsilon*100))
        ax[0].plot(np.array(ct)[:, 0], linewidth=2)
        ax[0].plot(np.array(ct)[:, 1])
        ax[0].plot([1]*len(ct))
                
        if len(reward_history) > 1 and next_ep:
            print("new_ax1")
            ax[1].clear()
            h = np.array(reward_history)
            hist = min(len(h), 20)
            x = np.arange(len(h)-hist, len(h))
            slope, intercept, _, _, _ = stats.linregress(x, h[-hist:])
            line = slope*x+intercept
            ax[1].plot(h)
            ax[1].plot(q_history)
            ax[1].plot(x, line)
            next_ep = False

        ax[0].set_autoscaley_on(False)
        ax[0].set_ylim([-1,2])
        ax[1].set_autoscaley_on(False)
        ax[1].set_ylim([-200,0])
        
        ct.clear()
    ct.append(np.array([task._system.x, task._control]))

# don't exit yet
plt.show()