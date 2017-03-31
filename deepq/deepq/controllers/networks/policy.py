from .builder import NetworkBuilder
import tensorflow as tf


class PolicyNet(object):
    def __init__(self, state, action, scope):
        self._action = action
        self._state  = state
        self._scope  = scope

    @property
    def action(self):
        return self._action

    @property
    def state(self):
        return self._state

    @property
    def scope(self):
        return self._scope

class ContinuousPolicyBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, features):
        super(ContinuousPolicyBuilder, self).__init__(state_size     = state_size, 
                                                      history_length = history_length,
                                                      num_actions    = num_actions)
        self._features = features

    def _build(self, inputs):
        state = inputs.get("state", None)
        if state is None:
            state = self.make_state_input("state")
        features = self._features(state)
        action   = tf.layers.dense(features, self.num_actions, activation=tf.tanh, name="action")
        return PolicyNet(state = state, action = action, scope = tf.get_variable_scope())

class GreedyPolicyBuilder(NetworkBuilder):
    def __init__(self, q_builder):
        super(GreedyPolicyBuilder, self).__init__(state_size     = q_builder.state_size, 
                                                  history_length = q_builder.history_length,
                                                  num_actions    = q_builder.num_actions)
        self._q_builder = q_builder

    def _build(self, inputs):
        pb = self._q_builder.build(reuse=True, inputs = inputs)
        proposed_actions = pb.q_values
        best_action = tf.argmax(proposed_actions, axis=1, name="best_action")
        return PolicyNet(action = best_action, state = pb.state, scope = tf.get_variable_scope())