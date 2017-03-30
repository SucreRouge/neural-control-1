from .builder import NetworkBuilder
import tensorflow as tf

class ContinuousQNetwork(object):
    def __init__(self, state, action, q_value, scope, summaries):
        self._action    = action
        self._state     = state
        self._q_value   = q_value
        self._scope     = scope
        self._summaries = summaries

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def q_value(self):
        return self._q_value

    @property
    def scope(self):
        return self._scope

    @property
    def summaries(self):
        return self._summaries


class ContinuousQBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, state_features, full_features):
        super(DiscreteQBuilder, self).__init__(state_size     = state_size, 
                                               history_length = history_length,
                                               num_actions    = num_actions)
        self._state_features = state_features
        self._full_features  = full_features

    def _build(self, state = None, action = None):
        state  = self.make_state_input("state") if state is None else state
        action = self.make_action_input("action") if action is None else action

        state_features = self._state_features(state)
        full_data = tf.concat([state_features, action], axis=1)
        full_features  = self._full_features(full_data)

        q_value = tf.layers.dense(full_features, 1, name="q_value")
        
        self._summaries.append(tf.summary.histogram("q_value", q_value))
        scope = tf.get_variable_scope()

        return ContinuousQNetwork(state   = state,   action = action, 
                                q_value   = q_value, scope  = scope, 
                                summaries = self._summaries)
        
