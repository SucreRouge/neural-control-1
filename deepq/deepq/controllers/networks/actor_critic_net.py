from ..utils import *
import tensorflow as tf
import numpy as np
from . import ContinuousQBuilder, NetworkBuilder, ContinuousPolicyBuilder, ContinuousBellmanBuilder

class ActorCriticNet(object):
    def __init__(self, global_step):
        self._global_step = global_step

    @property
    def global_step(self):
        return self._global_step

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
    def __init__(self, state_size, history_length, num_actions, policy_net, critic_net, 
                 soft_target_update=1e-4, discount=0.99, 
                 critic_regularizer=None, critic_init=None, policy_init=None):
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
            with tf.name_scope("regularization_loss"):
                critic_reg_loss = tf.reduce_sum(critic._regularizers)
            loss = mse_loss + critic_reg_loss

            # get all further gradients
            critic_grads = critic_optimizer.compute_gradients(loss)
            critic_grads = [u for u in critic_grads if u[0] is not None]
            cgops = tf.group(*[u[0] for u in critic_grads], name="all_critic_gradients")

        with tf.variable_scope("policy_training"):
            tvars = tf.trainable_variables()
            # Policy Gradient update of policy
            grad_a = tf.identity(tf.gradients(policy_critic.q_value, [policy_critic.action], name="action_gradient")[0], name="dQ_da")
            with tf.name_scope("policy_gradient"):
                # note the minus here: we want to increase the expected return, so we do gradient ascent!
                pgrads = tf.gradients(policy.action, tvars, -grad_a, name="policy_gradient")

            with tf.name_scope("regularizer_gradient"):
                policy_reg_loss = tf.reduce_sum(policy._regularizers)
                reggrads = tf.gradients(policy_reg_loss, tvars, name="reg_gradient")

            with tf.name_scope("combine_gradients"):
                with tf.name_scope("batch_size"):
                    batch_size = tf.to_float(tf.shape(policy_critic.action)[0])
                summed = list(map(_safe_add, zip(pgrads, reggrads)))
                for i in range(len(summed)):
                    if summed[i] is not None:
                        summed[i] = summed[i] / batch_size

            policy_grads = [u for u in zip(summed, tvars) if u[0] is not None]
            agops = tf.group(*[u for u in summed if u is not None], name="all_policy_gradients")

        # then perform the update. set control dependencies to ensure that weight changes are not 
        # interleaved with gradient calculations
        with tf.name_scope("train_step"):
            with tf.control_dependencies([agops, cgops]):
                ctrain = critic_optimizer.apply_gradients(critic_grads, global_step=self._net._global_step, name="CriticOptimizer")
                atrain = actor_optimizer.apply_gradients(zip(summed, tvars), name="PolicyOptimizer")
                train = tf.group(ctrain, atrain, name="train_step")


        with tf.name_scope("train_summary"):
            self._summaries.append(tf.summary.scalar("loss", loss))
            self._summaries.append(tf.summary.scalar("q_error", mse_loss))
            self._summaries.append(tf.summary.scalar("critic_regularizer", critic_reg_loss))
            self._summaries.append(tf.summary.scalar("policy_regularizer", policy_reg_loss))
            self._summaries.append(tf.summary.histogram("policy_gradient", grad_a))
            self._summaries.append(tf.summary.histogram("q_update", old_target_q - current_q))
            self._summaries.append(tf.summary.histogram("clipped_q_update", target_q - current_q))
            self._summaries.append(tf.summary.scalar("critic_lr", critic_optimizer._lr))
            self._summaries.append(tf.summary.scalar("policy_lr", actor_optimizer._lr))

        with tf.name_scope("gradient_summary"):
            for g, v in critic_grads:
                self._summaries.append(tf.summary.histogram(v.name, g))
            for g, v in policy_grads:
                self._summaries.append(tf.summary.histogram(v.name, g))
        
        
        self._net.set_training_ops(loss = loss, train = train)

def _safe_add(ts):
    t1, t2 = ts
    if t1 is not None:
        if t2 is not None:
            return t1 + t2
        else:
            return t1
    elif t2 is not None:
        return t2
    else:
        return None