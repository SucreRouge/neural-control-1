from ..utils import *
import tensorflow as tf
import numpy as np
from itertools import chain
from . import ContinuousQBuilder, NetworkBuilder, ContinuousPolicyBuilder, ContinuousBellmanBuilder

class ActorCriticNet(object):
    def __init__(self, global_step):
        self._global_step = global_step

    @property
    def global_step(self):
        return self._global_step

    def set_policy(self, policy):
        self._policy = policy

    def set_critic(self, critic):
        self._critic = critic

    @property
    def critic(self):
        return self._critic

    def set_bellman(self, bellman):
        self._bellman = bellman

    def set_training_ops(self, loss, train, train_critic):
        self._train_loss     = loss
        self._train_step     = train
        self._train_critic   = train_critic

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
        crt = self._critic.q_value
        if summary_writer is None:
            _, loss, Q = session.run([self._train_step, self._train_loss, crt], feed_dict = feed)
        else:
            _, loss, smr, step, Q = session.run([self._train_step, self._train_loss, self._summaries, self._global_step, crt], feed_dict = feed)
            summary_writer.add_summary(smr, step)
        return loss, Q

    def train_step_critic(self, qs, session, summary_writer=None):
        feed = self._train_feed(qs)
        crt = self._critic.q_value
        if summary_writer is None:
            _, loss, Q = session.run([self._train_critic, self._train_loss, crt], feed_dict = feed)
        else:
            _, loss, smr, step, Q = session.run([self._train_critic, self._train_loss, self._summaries, self._global_step, crt], feed_dict = feed)
            summary_writer.add_summary(smr, step)
        return loss, Q

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
    def __init__(self, state_size, history_length, num_actions, policy_net, critic_net, 
                 soft_target_update=1e-4, discount=0.99, 
                 critic_regularizer=None, critic_init=None, policy_init=None,
                 track_gradients=False, clip_critic_gradients=20.0, clip_policy_gradients=2.0):
        super(ActorCriticBuilder, self).__init__(state_size     = state_size, 
                                                 history_length = history_length, 
                                                 num_actions    = num_actions)

        self._soft_target_update = soft_target_update
        self._discount           = discount

        self._critic_builder = ContinuousQBuilder(state_size, history_length, num_actions, critic_net, 
                                                  regularizer = critic_regularizer,
                                                  initializer = critic_init)
        self._policy_builder = ContinuousPolicyBuilder(state_size, history_length, num_actions, policy_net,
                                                  initializer = policy_init)
        self._track_gradients = track_gradients
        self._clip_critic_gradients = clip_critic_gradients
        self._clip_policy_gradients = clip_policy_gradients

    def _build(self, actor_optimizer, critic_optimizer, inputs=None, gstep=None):
        if gstep is None:
            gstep    = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        discount = tf.Variable(self._discount, dtype=tf.float32, trainable=False, name='discount')
        if self._soft_target_update:
            tau = tf.Variable(self._soft_target_update, dtype = tf.float32, trainable=False, name="tau")

        self._net = ActorCriticNet(gstep)

        reward   = tf.placeholder(tf.float32, [None], name="reward")
        terminal = tf.placeholder(tf.bool,    [None], name="terminal")
        nstate  = self.make_state_input(name="next_state")

        state  = self.make_state_input()
        action = self.make_action_input()
        v = self._critic_builder.build(name_scope="critic", var_scope="critic", inputs={"state": state, "action": action})
        self._summaries += v.summaries
        self._net.set_critic(v)

        # target critic vars
        critic_target_scope = copy_variables_to_scope(v.scope, "target_vars/critic")
        if self._soft_target_update:
            update_critic = update_from_scope(v.scope, critic_target_scope, tau, "update_critic")
        else:
            update_critic = assign_from_scope(v.scope, critic_target_scope, "update_critic")

        p = self._policy_builder.build(name_scope="policy", var_scope="policy", inputs={"state": state})
        self._summaries += p.summaries
        self._net.set_policy(p)

        # target policy vars
        policy_target_scope = copy_variables_to_scope(p.scope, "target_vars/policy")
        if self._soft_target_update:
            update_policy = update_from_scope(p.scope, policy_target_scope, tau, "update_policy")
        else:
            update_policy = assign_from_scope(p.scope, policy_target_scope, "update_policy")

        self._net.set_update(actor = update_policy, critic = update_critic)
        
        # Bellman update of the critic
        bb = ContinuousBellmanBuilder(history_length = self.history_length, state_size = self.state_size, 
                                    num_actions = self.num_actions)
        inputs = {"discount": discount, "reward": reward, "terminal": terminal, "action": action,
                  "next_state": nstate}
        b = bb.build(qbuilder = self._critic_builder, pbuilder = self._policy_builder, 
                     critic_target_scope = critic_target_scope, policy_target_scope=policy_target_scope, inputs = inputs)
        self._net.set_bellman(b)

        # calculate q value of actor
        ae = self._critic_builder.build(name_scope="actor_evaluation", var_scope="critic", reuse=True, inputs={"state": state, "action": p.action})
        # add an identity op to make the graph easier to look at
        tf.identity(ae._q_value, name="Q_s_pa")

        self._build_training(actor_optimizer, critic_optimizer, v, p, b.updated_q, ae)
        self._summaries += b.summaries
        self._net._summaries = tf.summary.merge(self._summaries)

        return self._net

    def _build_training(self, actor_optimizer, critic_optimizer, critic, policy, target_q, policy_critic):
        # prepare name scopes for the summary ops
        with tf.name_scope("gradient_summary") as gsns:
            self._gradient_summary_name_scope = gsns

        with tf.name_scope("train_summary") as tsns:
            self._train_summary_name_scope = tsns


        cgops, critic_grads, loss = self._build_critic_gradients(critic_optimizer, critic, target_q)
        agops, policy_grads = self._build_actor_training(actor_optimizer, policy, policy_critic)

        # then perform the update. set control dependencies to ensure that weight changes are not 
        # interleaved with gradient calculations
        with tf.name_scope("train_step"):
            with tf.control_dependencies([agops, cgops]):
                # we do two optimization steps here. Only one should increase the global_step,
                # so we pass global_step only to the critic optimizer.
                ctrain = critic_optimizer.apply_gradients(critic_grads, global_step=self._net._global_step, name="CriticOptimizer")
                atrain = actor_optimizer.apply_gradients(policy_grads, name="PolicyOptimizer")
                train = tf.group(ctrain, atrain, name="train_step")

        with tf.name_scope(tsns):
            # if we assume that the Q function is correct, this tells us how much 
            # the polic has improved compared to steps in the replay buffer
            self._summaries.append(tf.summary.scalar("improvement", tf.reduce_mean(policy_critic.q_value - critic.q_value)))

        
        
        self._net.set_training_ops(loss = loss, train = train, train_critic = ctrain)

    def _build_critic_gradients(self, critic_optimizer, critic, target_q):
        """ 
            This function calculates the gradients that are required for critic training.
        """
        current_q = critic.q_value
        # calculate all the required gradients
        with tf.variable_scope("critic_training"):
            # ensure target_q is never too far from current_q
            with tf.name_scope("clipped_target"):
                bound    = 1
                old_target_q = target_q
                distance = tf.clip_by_value(target_q - current_q, -bound, bound)
                target_q = current_q + distance

            mse_loss = tf.losses.mean_squared_error(current_q, tf.stop_gradient(target_q), scope='loss')

            # regularization
            with tf.name_scope("regularization"):
                reg_loss = tf.reduce_sum(critic._regularizers, name="loss")
                reg_grads = critic_optimizer.compute_gradients(reg_loss)

            # calculate critic gradients and make an op that waits for all gradients to be computed
            grads = critic_optimizer.compute_gradients(mse_loss)
            with tf.name_scope("all_gradients"):
                cgops = tf.group(*[u[0] for u in chain(grads, reg_grads) if u[0] is not None], name="all_gradients")
            
            # apply clipping to the gradients
            clipped, cgnorm = self._get_clipped_gradients(grads, self._clip_critic_gradients)

            # now combine gradients
            with tf.name_scope("combine_gradients"):
                grads = apply_binary_op([c[0] for c in clipped], [g[0] for g in reg_grads], tf.add)
                grads = list(zip(grads, tf.trainable_variables()))

            loss = mse_loss + reg_loss

            

        with tf.name_scope(self._train_summary_name_scope):
            self._summaries.append(tf.summary.scalar("loss", loss))
            self._summaries.append(tf.summary.scalar("q_error", mse_loss))
            self._summaries.append(tf.summary.scalar("critic_regularizer", reg_loss))
            self._summaries.append(tf.summary.histogram("td_error", old_target_q - current_q))
            self._summaries.append(tf.summary.scalar("mean_td_error", tf.reduce_mean(old_target_q - current_q)))
            self._summaries.append(tf.summary.histogram("clipped_td_error", target_q - current_q))
            self._summaries.append(tf.summary.scalar("critic_gradient_norm", cgnorm))

        with tf.name_scope(self._gradient_summary_name_scope):
            summarize_gradients(grads, self._track_gradients)

        return cgops, grads, loss

    def _build_actor_training(self, actor_optimizer, policy, policy_critic):
        with tf.variable_scope("policy_training"):
            tvars = tf.trainable_variables()
            with tf.name_scope("batch_size"):
                batch_size = tf.to_float(tf.shape(policy_critic.action)[0])

            # Policy Gradient update of policy
            grad_a = tf.identity(tf.gradients(policy_critic.q_value, [policy_critic.action], name="action_gradient")[0], name="dQ_da")
            # note the minus here: we want to increase the expected return, so we do gradient ascent!
            grads = actor_optimizer.compute_gradients(policy.action, tvars, grad_loss=-grad_a / batch_size)

            # regularization
            with tf.name_scope("regularization"):
                reg_loss = tf.reduce_sum(policy._regularizers, name="loss")
                reg_grads = actor_optimizer.compute_gradients(reg_loss)

            with tf.name_scope("all_gradients"):
                agops = tf.group(*[u[0] for u in chain(grads, reg_grads) if u[0] is not None], name="all_gradients")

            
            clipped, norm = self._get_clipped_gradients(grads, self._clip_policy_gradients)
            # now combine gradients
            with tf.name_scope("combine_gradients"):
                grads = apply_binary_op([c[0] for c in clipped], [g[0] for g in reg_grads], tf.add)
                grads = list(zip(grads, tf.trainable_variables()))

        with tf.name_scope(self._train_summary_name_scope):
            self._summaries.append(tf.summary.scalar("policy_regularizer", reg_loss))
            self._summaries.append(tf.summary.histogram("policy_gradient", grad_a))
            self._summaries.append(tf.summary.scalar("policy_gradient_norm", norm))

        with tf.name_scope(self._gradient_summary_name_scope):
            summarize_gradients(grads, self._track_gradients)

        return agops, grads

    def _get_clipped_gradients(self, gradients_and_vars, norm):
        # ensure we get an iterable list
        gradients_and_vars = list(gradients_and_vars)

        # if no gradient clipping is set, this does nothing
        if norm is None:
            return gradients_and_vars

        gradients = [u[0] for u in gradients_and_vars]
        variables = [u[1] for u in gradients_and_vars]
        clipped, norm = tf.clip_by_global_norm(gradients, clip_norm = norm)

        return list(zip(clipped, variables)), norm