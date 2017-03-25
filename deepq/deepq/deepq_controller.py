import numpy as np
import tensorflow as tf
from .memory import History, Memory
from .qnet import QNet

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


class DeepQController(object):
    def __init__(self, history_length, memory_size, state_size, num_actions,
                    steps_per_epoch=10000, final_exploration_frame=1000000,
                    final_epsilon=0.1, minibatch_size=64):
        # configuration variables (these remain constant)
        self._num_actions     = num_actions
        self._state_size      = state_size
        self._history_length  = history_length
        self._steps_per_epoch = steps_per_epoch
        self._next_epoch      = None
        self._policy          = EGreedy(1.0, final_epsilon, final_exploration_frame)
        self._minibatch_size  = minibatch_size

        self._history         = History(duration=history_length, state_size=state_size)
        self._state_memory    = Memory(size=int(memory_size), history_length=history_length, state_size=state_size)
        self._session         = None
        self._last_action     = None

        # counters
        self._action_counter  = 0
        self._step_counter    = 0
        self._epoch_counter   = 0

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

    def train(self):
        if len(self._state_memory) < 10000:
            return

        summary_writer = self._summary_writer if self._step_counter % 100 == 0 else None

        sample = self._state_memory.sample(self._minibatch_size)
        ls = self._qnet.train_step(sample, self._session, summary_writer)
        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            # copy target net to policy net
            self._qnet.update_target(self._session)
            self._step_counter = 0
            self._epoch_counter += 1
            if self._next_epoch:
                self._next_epoch()

    def setup_graph(self, arch, target_net=True, double_q=False, dueling=False, learning_rate=1e-4):
        qnet = QNet(state_size     = self._state_size, 
                    history_length = self._history_length, 
                    num_actions    = self._num_actions,
                    double_q       = double_q,
                    target_net     = target_net,
                    dueling        = dueling)

        # TODO Figure these out!
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=0.01, momentum=0.95)
        self._qnet = qnet.build_graph(arch, opt)

    def init(self, session, logger):
        self._session = session
        self._session.run([tf.global_variables_initializer()])
        self._qnet.update_target(self._session)
        self._summary_writer = logger
