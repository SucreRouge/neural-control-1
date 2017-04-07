import tensorflow as tf
from .builder import NetworkBuilder

class DiscreteQNetwork(object):
    def __init__(self, state, q_values, scope, summaries):
        self._state     = state
        self._q_values  = q_values
        self._scope     = scope
        self._summaries = summaries

    @property
    def state(self):
        return self._state

    @property
    def q_values(self):
        return self._q_values

    @property
    def scope(self):
        return self._scope

    @property
    def summaries(self):
        return self._summaries


class DiscreteQBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, features, dueling = False):
        super(DiscreteQBuilder, self).__init__(state_size     = state_size, 
                                               history_length = history_length,
                                               num_actions    = num_actions)
        self._features     = features
        self._dueling_arch = dueling

    @property
    def dueling_arch(self):
        return self._dueling_arch

    def _build(self, inputs):
        state = inputs.get("state", None)
        if state is None:
            state = self.make_state_input("state")

        if self.dueling_arch:
            q_values = self._make_dueling_arch(state)
        else:
            q_values = self._make_single_arch(state)
        
        self._summaries.append(tf.summary.histogram("q_values", q_values))
        scope = tf.get_variable_scope()

        return DiscreteQNetwork(state = state, q_values  = q_values, 
                                scope = scope, summaries = self._summaries)

    def _make_single_arch(self, state):
        features = self._features(state)
        return tf.layers.dense(features, self.num_actions, name="q_values")

    def _make_dueling_arch(self, state):
        features = self._features(state)
        action_vals = tf.layers.dense(features, self.num_actions, name="action_values")
        state_vals  = tf.layers.dense(features, 1, name="state_values")
        with tf.name_scope("q_values"):
            q_vals = state_vals + action_vals - tf.reduce_mean(action_vals, axis=1, keep_dims=True)

        self._summaries.append(tf.summary.histogram("action_values", action_vals))
        self._summaries.append(tf.summary.histogram("state_values",  state_vals))

        return q_vals