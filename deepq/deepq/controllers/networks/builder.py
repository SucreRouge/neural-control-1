import tensorflow as tf
from ..utils import assign_from_scope

# graph builder base class with utilities
class NetworkBuilder(object):
    def __init__(self, state_size, history_length, num_actions):
        # TODO mazbe make graph a build param instead
        self._state_size     = state_size
        self._history_length = history_length
        self._num_actions    = num_actions
        self._summaries      = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def history_length(self):
        return self._history_length

    @property
    def num_actions(self):
        return self._num_actions

    def make_state_input(self, name="state"):
        return tf.placeholder(tf.float32, [None, self.history_length, self.state_size], name=name)

    def make_action_input(self, name="action"):
        # TODO generalize this function so that it handles both discrete and continuous cases gracefully
        return tf.placeholder(tf.float32, [None, self.num_actions], name=name)

    def build(self, graph = None, name_scope = None, var_scope = None, reuse = None, inputs=None, **kwargs):
        inputs = inputs if inputs is not None else {} 
        graph  = graph  if graph  is not None else tf.get_default_graph()

        with graph.as_default():
            if name_scope is not None:
                name_scope = tf.name_scope(name_scope).__enter__()

            if var_scope is not None:
                var_scope = tf.variable_scope(var_scope).__enter__()
            else:
                var_scope = tf.get_variable_scope()

            self._summaries = []
            with tf.variable_scope(var_scope, reuse = reuse):
                if name_scope is not None:
                    with tf.name_scope(name_scope):
                        net = self._build(inputs = inputs, **kwargs)
                elif var_scope.name != "":
                    with tf.name_scope(var_scope.name + "/"):
                        net = self._build(inputs = inputs, **kwargs)
                else:
                    net = self._build(inputs = inputs, **kwargs)
            self._summaries = None
            return net
    