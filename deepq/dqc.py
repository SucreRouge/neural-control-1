import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq.memory import *

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

class QState(object):
    def __init__(self, current, next, reward, action, terminal = False):
        self.current  = current
        self.next     = next
        self.reward   = reward
        self.action   = action
        self.terminal = terminal


class QNet(object):
    def __init__(self, current, next, chosen, loss, train, reward, terminal, qvals):
        self.current  = current
        self.next     = next
        self.chosen   = chosen
        self.loss     = loss
        self.train    = train
        self.reward   = reward
        self.terminal = terminal
        self.qvals    = qvals

    def train_step(self, qs, session):
        feed = {self.current:  qs.current,
                self.next:     qs.next,
                self.chosen:   qs.action,
                self.reward:   qs.reward,
                self.terminal: qs.terminal}

        _, loss = session.run([self.train, self.loss], feed_dict = feed)
        return loss

    def actions(self, state, session):
        return session.run([self.qvals], feed_dict={self.current:state})[0]


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
    def __init__(self, history_length, memory_size, task, session):
        self._num_actions    = task.num_actions
        self._state_size     = task.state_size
        self._history_length = history_length
        self._history        = History(duration=history_length, state_size=task.state_size)
        self._state_memory   = Memory(size=memory_size, history_length=history_length, state_size=task.state_size)
        self._task           = task
        self._session        = session
        self._policy         = EGreedy(1.0)
        self._do_training    = True
        self._step_counter   = 0
        self._epoch_q        = deque()
        self._epoch_rwd      = deque()
        self._epoch_counter  = 0
        self._rwd_hist       = deque()
        self._steps_per_epoch = 10000
        self._history.observe(self._task.state)

    def step(self):
        # get the current state (including history). Fill with 0 if nothing is present.
        # TODO maybe fill with first value?
        full_state = self._history.state
        full_state = full_state[np.newaxis,:]

        action_vals   = self._qnet.actions(full_state, self._session)
        action        = self._policy(action_vals)
        self._epoch_q.append(np.max(action_vals))

        # observe what happens in task
        reward, terminal = self._task.update(action)
        self._epoch_rwd.append(reward)

        # gather data
        next_full = self._history.observe(self._task.state)
        next_full = next_full[np.newaxis,:]
        self._state_memory.append(state=full_state, next = None if terminal else next_full, reward=reward, action=action)

        # is the epoch finished?
        if terminal:
            self._history.clear()
            self._task.reset()

        # update policy epsilon
        frac_epoch = self._epoch_counter + self._step_counter / float(self._steps_per_epoch)
        self._policy.epsilon = 1 - frac_epoch / 100
        if self._policy.epsilon < 0.1:
            self._policy.epsilon = 0.1

        # do training
        if self._do_training and len(self._state_memory) > 1000 and self._task._stepcount % 4 == 0:
            self.train()

    def train(self):
        ls = self._qnet.train_step(self._state_memory.sample(32), self._session)
        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            print("mean Q: %f"%(np.mean(self._epoch_q)))
            # copy target net to policy net
            self._session.run(self._update_q_op)
            self._step_counter = 0
            self._epoch_q.clear()

            self._rwd_hist.append(np.mean(self._epoch_rwd))
            self._epoch_rwd.clear()
            self._epoch_counter += 1

    def setup_graph(self, arch):
        # the target net
        with tf.variable_scope("tnet"):
            qi, qa = build_q_net(self._task.state_size, self._history_length, self._num_actions, arch)

        # the policy net
        with tf.variable_scope("qnet"):
            ti, ta   = build_q_net(self._task.state_size, self._history_length, self._num_actions, arch)
            reward   = tf.placeholder(tf.float32, [None], name="reward")
            chosen   = tf.placeholder(tf.int32, [None], name="chosen")
            terminal = tf.placeholder(tf.bool, [None], name="chosen")
            discount = tf.Variable(0.99, dtype=tf.float32, trainable=False)

            best_future_q = tf.reduce_max(qa, axis=1)
            num_samples = tf.shape(chosen)[0]
            indices =  tf.transpose(tf.stack([tf.range(0, num_samples), chosen]))
            current_q = tf.gather_nd(ta, indices)
            state_value = best_future_q * discount * tf.to_float(terminal) + reward

            loss = tf.losses.mean_squared_error(current_q, tf.stop_gradient(state_value))
            train = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9).minimize(loss)

        # make the copy operation
        # copy target net to policy net
        qnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="qnet")
        tnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="tnet")
        asgns = []
        for qnet_var in qnet_vars:
            for tnet_var in tnet_vars:
                if qnet_var.name[4:] == tnet_var.name[4:]:
                    asgns.append(tnet_var.assign(qnet_var))
        self._update_q_op = tf.group(*asgns)

        self._qnet = QNet(current = ti, next = qi, chosen = chosen, 
                               loss = loss, train = train, reward = reward,
                               terminal = terminal, qvals=ta)


def build_q_net(state_size, history_length, num_actions, arch):
    inp = tf.placeholder(tf.float32, [None, history_length, state_size], name="state")
    features = arch(inp)
    actions  = tf.layers.dense(features, num_actions, name="qvalues")

    return inp, tf.reshape(actions, [-1, num_actions])

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

ql = QLearner(history_length=10, memory_size=50000, task=task, session=tf.Session())
ql.setup_graph(arch)
ql._session.run([tf.global_variables_initializer()])
ct = deque()
fig, ax = plt.subplots(2,1)
for i in range(10000000):
    ql.step()
    if task._stepcount == 0:
        ax[0].clear()
        ax[0].set_title("Epoch: %d , Epsilon=%f"%(ql._epoch_counter, ql._policy.epsilon))
        ax[0].plot(np.array(ct)[:, 0], linewidth=2)
        ax[0].plot(np.array(ct)[:, 1])
        ax[0].plot([1]*len(ct))
        plt.pause(0.001)

        if len(ql._rwd_hist) > 1:
            ax[1].clear()
            h = np.array(ql._rwd_hist)
            hist = min(len(h), 25)
            x = np.arange(len(h)-hist, len(h))
            slope, intercept, _, _, _ = stats.linregress(x, h[-hist:])
            line = slope*x+intercept
            ax[1].plot(h)
            ax[1].plot(x, line)

        ct.clear()
    ct.append(np.array([task._system.x, task._control]))

# don't exit yet
plt.show()