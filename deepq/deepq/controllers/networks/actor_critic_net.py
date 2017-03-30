from .util import *
import tensorflow as tf

class PolicyGraph(object):
    def __init__(self):
        pass

class CriticGraph(object):
    def __init__(self):
        pass

class ActorCritic(object):
    def __init__(self, state_size, history_length):
        super(ActorCritic, self).__init__(state_size = state_size, history_length = history_length)

    def build(self):
        # build the critic
        
        pass

    def _build_q_net(self, input, actions, state_arch, q_arch):
        history_length = input.get_shape()[1].value
        state_size     = input.get_shape()[2].value
        
        processed_state = arch(input)
        combined = tf.concat([processed_state, actions], axis=1 )
        
        features = q_arch(combined)
        qvalues  = tf.layers.dense(features, 1, name="qvalues")
        return qvalues