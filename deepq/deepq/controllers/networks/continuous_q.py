import tensorflow as tf
from .builder import NetworkBuilder

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

    def critique(self, state, action, session):
        feed = {self.state: state, self.action: action}
        return session.run(self.q_value, feed_dict = feed)


class ContinuousQBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, features, 
                 regularizer = None, initializer = None):
        super(ContinuousQBuilder, self).__init__(state_size     = state_size, 
                                                 history_length = history_length,
                                                 num_actions    = num_actions)
        self._features    = features
        self._regularizer = regularizer
        self._initializer = initializer

    def _build(self, inputs):
        state = inputs.get("state", None)
        if state is None:
            state = self.make_state_input("state")
        
        action = inputs.get("action", None)
        if action is None:
            action = self.make_action_input("action")

        features = self._features(state, action)
        q_value  = tf.layers.dense(features, 1, name="q_value", 
                                   kernel_regularizer = self._regularizer,
                                   kernel_initializer = self._initializer)
        
        with tf.name_scope("summary"):
            self._summaries.append(tf.summary.scalar("mean_q", tf.reduce_mean(q_value)))
        scope = tf.get_variable_scope()

        return ContinuousQNetwork(state   = state,   action = action, 
                                q_value   = q_value, scope  = scope, 
                                summaries = self._summaries)
        
