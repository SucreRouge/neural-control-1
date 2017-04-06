from .builder import NetworkBuilder
import tensorflow as tf
from ..utils import *

class BellmanNet(object):
    def __init__(self, updated_q, reward, terminal, action, state, summaries):
        self._updated_q = updated_q
        self._reward    = reward
        self._terminal  = terminal
        self._action    = action
        self._state     = state
        self._summaries = summaries

    @property
    def updated_q(self):
        return self._updated_q

    @property
    def reward(self):
        return self._reward

    @property
    def action(self):
        return self._action

    @property
    def terminal(self):
        return self._terminal

    @property
    def state(self):
        return self._state

    @property
    def summaries(self):
        return self._summaries

class ContinuousBellmanBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions):
        super(ContinuousBellmanBuilder, self).__init__(state_size     = state_size,
                                                       history_length = history_length,
                                                       num_actions    = num_actions)

    def _build(self, inputs, qbuilder, pbuilder, critic_target_scope, policy_target_scope):
        discount = inputs['discount']
        reward   = inputs['reward']
        chosen   = inputs['action']
        terminal = inputs['terminal']
        next_state = inputs['next_state']

        with tf.name_scope("future_action"):
            next_action = pbuilder.build(var_scope = policy_target_scope, name_scope = current_name_scope(), 
                                     reuse=True, inputs={"state": next_state})
            next_action = next_action.action

        with tf.name_scope("future_return"):
            next_q = qbuilder.build(var_scope = critic_target_scope, name_scope = current_name_scope(), 
                                     reuse=True, inputs={"state": next_state, "action": next_action})
            self._summaries += next_q.summaries
            future_return = next_q.q_value * (1.0 - tf.to_float(terminal))

        with tf.name_scope("discounted_return"):
            target_q = discount * future_return + reward

        return BellmanNet(updated_q = target_q, reward = reward, action = chosen, terminal = terminal, 
                          state = next_state, summaries = self._summaries)


class DiscreteBellmanBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, double_q = True):
        super(DiscreteBellmanBuilder, self).__init__(state_size     = state_size,
                                                     history_length = history_length,
                                                     num_actions    = num_actions)
        self._double_q = double_q

    @property
    def double_q(self):
        return self._double_q

    def _build(self, inputs, qbuilder, value_scope, target_scope):
        discount = inputs['discount']
        reward   = inputs['reward']
        chosen   = inputs['action']
        terminal = inputs['terminal']
        next_state = inputs['next_state']
        state    = inputs['state']

        # calcuate Qs of the next state
        with tf.name_scope("future_q"):
            next_qs = qbuilder.build(var_scope = target_scope, name_scope = current_name_scope(), 
                                     reuse=True, inputs={"state": next_state})
            self._summaries += next_qs.summaries

        with tf.name_scope("future_return"):
            future_q = self._get_future_q(qbuilder, value_scope, next_qs.q_values, state)
            future_return = future_q * (1.0 - tf.to_float(terminal))

        with tf.name_scope("discounted_return"):
            target_q = discount * future_return + reward

        return BellmanNet(updated_q = target_q, reward = reward, action = chosen, terminal = terminal, 
                          state = next_state, summaries = self._summaries)

    def _get_future_q(self, qbuilder, value_scope, next_qs, state):
        if self.double_q:
            with tf.name_scope("best_action") as nscope:
                pb = qbuilder.build(var_scope=value_scope, name_scope=nscope, reuse=True, inputs={"state": state})
                self._summaries += pb.summaries
                proposed_actions = pb.q_values
                best_action = tf.argmax(proposed_actions, axis=1, name="best_action")
            return choose_from_array(next_qs, best_action, name="future_Q")
        else:
            return tf.reduce_max(next_qs, axis=1, name="future_Q")
