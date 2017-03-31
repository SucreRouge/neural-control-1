from ..utils import *
import tensorflow as tf
import numpy as np
from . import ContinuousQBuilder, NetworkBuilder, ContinuousPolicyBuilder, ContinuousBellmanBuilder

class ActorCriticNet(object):
    def __init__(self, global_step):
        self._global_step = global_step
        self._summaries   = tf.no_op()

    def set_policy(self, policy):
        self._policy = policy

    def set_bellman(self, bellman):
        self._bellman = bellman

    def set_training_ops(self, loss, train):
        self._train_loss     = loss
        self._train_step     = train

    def _train_feed(self, qs):
        b = self._bellman
        feed = {self._policy.state:  qs.current,
                b.state:    qs.next,
                b.action:   qs.action,
                b.reward:   qs.reward,
                b.terminal: qs.terminal}
        return feed

    def train_step(self, qs, session, summary_writer=None):
        feed = self._train_feed(qs)
        if summary_writer is None:
            _, loss = session.run([self._train_step, self._train_loss], feed_dict = feed)
        else:
            _, loss, smr, step = session.run([self._train_step, self._train_loss, self._summaries, self._global_step], feed_dict = feed)
            summary_writer.add_summary(smr, step)
        return loss

    def set_update(self, actor, critic):
        self._actor_update  = actor
        self._critic_update = critic

    def update_target(self, session):
        session.run([self._critic_update, self._actor_update])

    def get_actions(self, state, session):
        # batchify state
        state = state[np.newaxis, :]
        return session.run([self._policy.action], feed_dict={self._policy.state:state})[0][0]

class ActorCriticBuilder(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, state_features, full_features, 
                 target_critic = True, target_policy = True, soft_target_update=1e-4):
        super(ActorCriticBuilder, self).__init__(state_size     = state_size, 
                                                 history_length = history_length, 
                                                 num_actions    = num_actions)

        self._use_target_critic  = target_critic
        self._use_target_policy  = target_policy
        self._soft_target_update = soft_target_update

        self._critic_builder = ContinuousQBuilder(state_size, history_length, num_actions, state_features, full_features)
        self._policy_builder = ContinuousPolicyBuilder(state_size, history_length, num_actions, state_features)

    def _build(self, optimizer, inputs=None):
        gstep    = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        discount = tf.Variable(0.99, dtype=tf.float32, trainable=False, name='discount')
        if self._soft_target_update:
            tau = tf.Variable(self._soft_target_update, dtype = tf.float32, trainable=False, name="tau")

        self._net = ActorCriticNet(gstep)

        #with tf.name_scope("transition"):
        if True:
            reward   = tf.placeholder(tf.float32, [None], name="reward")
            terminal = tf.placeholder(tf.bool,    [None], name="terminal")
            nstate  = self.make_state_input(name="next_state")

        state  = self.make_state_input()
        action = self.make_action_input()
        v = self._critic_builder.build(name_scope="critic", var_scope="critic", inputs={"state": state, "action": action})
        self._summaries += v.summaries
        if self._use_target_critic:
            target_scope = copy_variables_to_scope(v.scope, "target_vars")
            with tf.name_scope(target_scope.name+"/"):
                if self._soft_target_update:
                    update_critic = update_from_scope(v.scope, target_scope, tau, "update_critic")
                else:
                    update_critic = assign_from_scope(v.scope, target_scope, "update_critic")
        else:
            target_scope = v.scope 
            update_critic = tf.no_op(name="update_critic")

        p = self._policy_builder.build(name_scope="policy", var_scope="policy", inputs={"state": state})
        self._net.set_policy(p)
        if self._use_target_policy:
            target_scope = copy_variables_to_scope(p.scope, "target_vars")
            with tf.name_scope(target_scope.name+"/"):
                if self._soft_target_update:
                    update_policy = update_from_scope(v.scope, target_scope, tau, "update_policy")
                else:
                    update_policy = assign_from_scope(p.scope, target_scope, "update_policy")
        else:
            target_scope = p.scope 
            update_policy = tf.no_op(name="update_policy")

        self._net.set_update(actor = update_policy, critic = update_critic)
        
        # Bellman update of the critic
        bb = ContinuousBellmanBuilder(history_length = self.history_length, state_size = self.state_size, 
                                    num_actions = self.num_actions)
        inputs = {"discount": discount, "reward": reward, "terminal": terminal, "action": action,
                  "next_state": nstate, "next_action": p.action}
        b = bb.build(qbuilder = self._critic_builder, value_scope = v.scope, 
                     target_scope = target_scope, inputs = inputs)
        self._net.set_bellman(b)

        self._build_training(optimizer, v, p, b.updated_q)
        self._summaries += b.summaries
        self._net._summaries = tf.summary.merge(self._summaries)

        return self._net

    def _build_training(self, optimizer, critic, policy, target_q):
        current_q = critic.q_value
        with tf.variable_scope("training"):
            with tf.name_scope("num_samples"):
                num_samples = tf.shape(current_q)[0]
            with tf.name_scope("scale"):
                scale = 1.0/tf.to_float(num_samples)
            loss    = tf.losses.mean_squared_error(current_q, tf.stop_gradient(target_q), scope='loss')
            # error clipping
            with tf.name_scope("clipped_error_gradient"):
                bound = scale
                q_error = tf.clip_by_value(tf.gradients(loss, [current_q])[0], -bound, bound)

            # get all further gradients
            tvars = tf.trainable_variables()
            tgrads = tf.gradients(current_q, tvars, q_error, "critic_gradients")
            def add_w_none(t):
                a, b = t
                if b is not None:
                    b = -b * scale
                if a is None:
                    return b 
                elif b is None:
                    return a
                return a + b

             # Policy Gradient update of policy
            with tf.name_scope("policy_gradient"):
                grad_a = tf.gradients(critic.q_value, [critic.action], name="dQ_da")
                pgrads = tf.gradients(policy.action, tf.trainable_variables(), grad_a)

            with tf.variable_scope("combined_gradients"):
                grads = map(add_w_none, zip(tgrads, pgrads))
                grads_and_vars = zip(grads, tvars)

            with tf.name_scope("summary"):
                self._summaries.append(tf.summary.scalar("loss", loss))
            train = optimizer.apply_gradients(grads_and_vars, global_step=self._net._global_step)
        
        self._net.set_training_ops(loss = loss, train = train)
