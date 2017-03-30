import numpy as np
import tensorflow as tf
from ..utils import *
from . import DiscreteQBuilder, NetworkBuilder

class QNetGraph(object):
    def __init__(self, global_step):
        self._summaries     = None
        self._global_step   = global_step

    def set_summaries(self, summaries):
        self._summaries = summaries

    def set_value_network(self, input, output, scope):
        self._value_in    = input
        self._value_out   = output
        self._value_scope = scope

    def set_target_network(self, input, output, update):
        self._target_in     = input
        self._target_out    = output
        self._target_update = update

    def set_bellman_ops(self, current, chosen, reward, next, terminal, current_q,
                        target_q):
        self._bellman_current  = current
        self._bellman_chosen   = chosen
        self._bellman_reward   = reward
        self._bellman_next     = next
        self._bellman_terminal = terminal
        self._bellman_currentQ = current_q
        self._bellman_targetQ  = target_q

    def set_training_ops(self, loss, train):
        self._train_loss     = loss
        self._train_step     = train

    def get_actions(self, state, session):
        # batchify state
        state = state[np.newaxis, :]
        return session.run([self._value_out], feed_dict={self._value_in:state})[0]

    def _train_feed(self, qs):
        actions = np.reshape(qs.action, (len(qs.action),))
        feed = {self._bellman_current:  qs.current,
                self._bellman_next:     qs.next,
                self._bellman_chosen:   actions,
                self._bellman_reward:   qs.reward,
                self._bellman_terminal: qs.terminal}
        return feed

    def train_step(self, qs, session, summary_writer=None):
        feed = self._train_feed(qs)
        if summary_writer is None:
            _, loss = session.run([self._train_step, self._train_loss], feed_dict = feed)
        else:
            _, loss, smr, step = session.run([self._train_step, self._train_loss, self._summaries, self._global_step], feed_dict = feed)
            summary_writer.add_summary(smr, step)
        return loss

    def update_target(self, session):
        session.run(self._target_update)

class QNet(NetworkBuilder):
    def __init__(self, state_size, history_length, num_actions, arch, 
                 target_net = True, double_q=False, dueling=False):
        super(QNet, self).__init__(state_size = state_size, history_length = history_length, num_actions = num_actions)

        self._double_q       = double_q
        self._use_target_net = target_net
        self._dueling_arch   = dueling
        self._summaries      = []
        self._q_builder      = DiscreteQBuilder(state_size, history_length, num_actions, arch, dueling = dueling)

    def _build(self, optimizer):
        self._summaries = []
        gstep    = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        discount = tf.Variable(0.99, dtype=tf.float32, trainable=False, name='discount')
        self._qnet  = QNetGraph(gstep)

        n = self._q_builder.build(var_scope='value_network', name_scope='value_network')
        self._qnet.set_value_network(input = n.state, output = n.q_values, scope = n.scope)
        target_vars = 'target_network' if self._use_target_net else 'value_network'
        n = self._q_builder.build(var_scope=target_vars, name_scope='target_network', reuse=not self._use_target_net)

        if self._use_target_net:
            update_target = assign_from_scope(self._qnet._value_scope, target_vars, name="update_target_network")
        else:
            update_target = tf.no_op()

        self._qnet.set_target_network(input = n.state, output = n.q_values, update = update_target)

        with tf.variable_scope("bellman_update"):
            current_q, updated_q = self._build_bellman_update(discount)

        self._qnet  = self._build_training(optimizer)
        self._qnet.set_summaries(tf.summary.merge(self._summaries))
        self._summaries = []

        return self._qnet

    def _build_bellman_update(self, discount):
        reward   = tf.placeholder(tf.float32, [None], name="reward")
        chosen   = tf.placeholder(tf.int32,   [None], name="chosen")
        terminal = tf.placeholder(tf.bool,    [None], name="terminal")

        target_input   = self._qnet._target_in
        value_input    = self._qnet._value_in
        target_actions = self._qnet._target_out
        value_actions  = self._qnet._value_out


        # TODO debug to check that this does what i think it does
        # target vaues
        with tf.name_scope("future_return"):
            if self._double_q:
                with tf.name_scope("best_action") as nscope:
                    pb = self._q_builder.build(var_scope=self._qnet._value_scope, name_scope=nscope, reuse=True)
                    proposed_actions = pb.q_values
                    best_action = tf.argmax(proposed_actions, axis=1, name="best_action")
                best_future_q = choose_from_array(target_actions, best_action, name="best_future_Q")
            else:
                best_future_q = tf.reduce_max(target_actions, axis=1, name="best_future_Q")                
            future_return = best_future_q * (1.0 - tf.to_float(terminal))

        with tf.name_scope("updated_Q"):
            target_q = discount * future_return + reward

        # current values
        with tf.name_scope("current_Q"):
            current_q     = choose_from_array(value_actions, chosen)
            self._summaries.append(tf.summary.scalar("mean_Q", tf.reduce_mean(current_q)))

        self._qnet.set_bellman_ops(current  = value_input, next   = target_input, 
                                   chosen   = chosen,      reward = reward,
                                   terminal = terminal,    current_q = current_q,
                                   target_q = target_q)

        return current_q, target_q

    def _build_training(self, optimizer):
        current_q = self._qnet._bellman_currentQ
        target_q  = self._qnet._bellman_targetQ
        with tf.variable_scope("training"):
            num_samples = tf.shape(current_q)[0]
            loss    = tf.losses.mean_squared_error(current_q, tf.stop_gradient(target_q), scope='loss')
            # error clipping
            with tf.name_scope("clipped_error_gradient"):
                bound = 1.0/tf.to_float(num_samples)
                q_error = tf.clip_by_value(tf.gradients(loss, [current_q])[0], -bound, bound)

            # get all further gradients
            tvars = tf.trainable_variables()
            tgrads = tf.gradients(current_q, tvars, q_error)
            grads_and_vars = zip(tgrads, tvars)

            self._summaries.append(tf.summary.scalar("loss", loss))
            train = optimizer.apply_gradients(grads_and_vars, global_step=self._qnet._global_step)
        
        self._qnet.set_training_ops(loss = loss, train = train)
        return self._qnet
