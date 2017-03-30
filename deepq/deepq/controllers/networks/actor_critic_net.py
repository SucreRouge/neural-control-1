from ..utils import *
import tensorflow as tf
from . import ContinuousQBuilder, NetworkBuilder, ContinuousPolicyBuilder

class ActorCriticNet(object):
    def __init__(self):
        pass

    def set_critics(self, value, target, update):
        self._critic = value 
        self._target_critic = target
        self._update_critic = update

class ActorCriticBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, state_features, full_features, 
                 target_critic = True, target_policy = True):
        super(ActorCriticBuilder, self).__init__(state_size     = state_size, 
                                                 history_length = history_length, 
                                                 num_actions    = num_actions)

        self._use_target_critic = target_critic
        self._use_target_policy = target_policy

        self._critic_builder = ContinuousQBuilder(state_size, history_length, num_actions, state_features, full_features)
        self._policy_builder = ContinuousPolicyBuilder(state_size, history_length, num_actions, state_features)

    def _build(self, inputs=None):
        gstep    = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        discount = tf.Variable(0.99, dtype=tf.float32, trainable=False, name='discount')

        self._net = ActorCriticNet()

        state  = self.make_state_input()
        action = self.make_action_input()
        inputs = {"state": state, "action": action}

        nstate  = self.make_state_input(name="next_state")
        tinputs = {"state": nstate}

        v, t, u = self._policy_builder.build_with_target(scope="policy", share = not self._use_target_policy, 
                                                         inputs = inputs, target_inputs = tinputs)

        tinputs["action"] = t.action
        v, t, u = self._critic_builder.build_with_target(scope="critic", share = not self._use_target_critic,
                                                         inputs = inputs, target_inputs = tinputs)
        self._net.set_critics(v, t, u)

        with tf.name_scope("bellman_update"):
            self._build_bellman_update_critic(discount)

    def _build_bellman_update_critic(self, discount):
        reward   = tf.placeholder(tf.float32, [None], name="reward")
        terminal = tf.placeholder(tf.bool,    [None], name="terminal")

        # TODO debug to check that this does what i think it does
        # target vaues
        with tf.name_scope("future_return"):
            future_return = self._net._target_critic.q_value * (1.0 - tf.to_float(terminal))

        with tf.name_scope("updated_Q"):
            target_q = discount * future_return + reward

        # current values
        with tf.name_scope("current_Q"):
            current_q     = self._net._critic.q_value
            self._summaries.append(tf.summary.scalar("mean_Q", tf.reduce_mean(current_q)))


        return current_q, target_q