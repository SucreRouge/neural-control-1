from .networks import ActorCriticBuilder
from .controller import Controller
from ..action_space import ActionSpace
from .memory import History, Memory
import tensorflow as tf
import numpy as np

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
            a = np.random.rand(*actions.shape) * 2 - 1
            return a
        else:
            return actions

class DeepPolicyGradientController(Controller):
    def __init__(self, history_length, memory_size, state_size, action_space,
                    steps_per_epoch=10000, minibatch_size=64, final_exploration_frame=1000000,
                    final_epsilon=0.1):
        action_space = ActionSpace(action_space)
        assert not action_space.is_discrete, "DeepPolicyGradientController works only on continuous action spaces"
        super(DeepPolicyGradientController, self).__init__(action_space, state_size, history_length)

        self._num_actions     = action_space.num_actions[0]
        self._steps_per_epoch = steps_per_epoch
        self._next_epoch      = None
        self._minibatch_size  = minibatch_size
        self._policy          = EGreedy(1.0, final_epsilon, final_exploration_frame)
        self._state_memory    = Memory(size=int(memory_size), history_length=history_length, state_size=state_size,
                                       action_dim = self._num_actions, action_type = float)

        # counters
        self._step_counter    = 0
        self._epoch_counter   = 0

    def _observe(self, state, last, reward, action, test=False):
        # if this is the first state, there is no transition to remember,
        # so simply add to the state history
        if action is None:
            return

        # if not in test mode, remember the transition
        if not test:
            self._state_memory.append(state=last, next=state, reward=reward, action=action)

    def _get_action(self, test=False):
        action_vals   = self._qnet.get_actions(self.full_state, self.session)
        action        = self._policy(action_vals, test)
        action        = np.clip(action, -1, 1)
        if not test:
            self._policy.set_stepcount(self.frame_count)
        
        return action, action_vals

    def train(self):
        if len(self._state_memory) < 10000:
            return

        summary_writer = self._summary_writer if self._step_counter % 100 == 0 else None

        sample = self._state_memory.sample(self._minibatch_size)
        ls = self._qnet.train_step(sample, self._session, summary_writer)
        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            # copy target net to policy net if we do hard target updates
            if not self._soft_target_update:
                self._qnet.update_target(self._session)
            self._step_counter = 0
            self._epoch_counter += 1
            if self._next_epoch:
                self._next_epoch()

    def setup_graph(self, state_features, action_features, full_features, graph = None, 
                          target_net=True, actor_learning_rate=1e-4, critic_learning_rate=1e-4, 
                          soft_target=False):
        qnet = ActorCriticBuilder(state_size     = self._state_size, 
                    history_length  = self.history_length, 
                    num_actions     = self._num_actions,
                    target_critic   = target_net,
                    target_policy   = target_net,
                    soft_target_update = soft_target,
                    state_features  = state_features,
                    action_features = action_features,
                    full_features   = full_features)

        self._soft_target_update = soft_target

        # TODO Figure these out!
        aopt = tf.train.AdamOptimizer(learning_rate=actor_learning_rate)
        copt = tf.train.AdamOptimizer(learning_rate=critic_learning_rate)
        self._qnet = qnet.build(actor_optimizer=aopt, critic_optimizer=copt, graph = graph)