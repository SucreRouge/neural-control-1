import numpy as np
import tensorflow as tf
from ..utils import *
from . import DiscreteQBuilder, NetworkBuilder, DiscreteBellmanBuilder

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

    def set_update(self, update):
        self._target_update = update

    def set_bellman_ops(self, bnet):
        self._bellman_chosen   = bnet.action
        self._bellman_reward   = bnet.reward
        self._bellman_next     = bnet.state
        self._bellman_terminal = bnet.terminal
        self._bellman_targetQ  = bnet.updated_q

    def set_training_ops(self, loss, train):
        self._train_loss     = loss
        self._train_step     = train

    def get_actions(self, state, session):
        # batchify state
        state = state[np.newaxis, :]
        return session.run([self._value_out], feed_dict={self._value_in:state})[0]

    def _train_feed(self, qs):
        actions = np.reshape(qs.action, (len(qs.action),))
        feed = {self._value_in:        qs.current,
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
        self._q_builder      = DiscreteQBuilder(state_size, history_length, num_actions, arch, dueling = dueling)

    def _build(self, optimizer, inputs=None):
        gstep    = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        discount = tf.Variable(0.99, dtype=tf.float32, trainable=False, name='discount')
        chosen   = tf.placeholder(tf.int32,[None], name="action")
        
        # grouping these so the graph visualization looks nicer
        with tf.name_scope("transition"):
            reward   = tf.placeholder(tf.float32, [None], name="reward")
            terminal = tf.placeholder(tf.bool,    [None], name="terminal")
            nstate  = self.make_state_input(name="next_state")

        self._qnet  = QNetGraph(gstep)

        state = self.make_state_input()
        v = self._q_builder.build(name_scope="qnet", var_scope="qnet", inputs={"state": state})
        self._qnet.set_value_network(input  = v.state, output = v.q_values, scope = v.scope)
        self._summaries += v.summaries

        if self._use_target_net:
            target_scope = copy_variables_to_scope(v.scope, "target_vars")
            with tf.name_scope(target_scope.name+"/"):
                update = assign_from_scope(v.scope, target_scope, "update")
        else:
            target_scope = v.scope 
            update = tf.no_op(name="update")
        self._qnet.set_update(update)

        bb = DiscreteBellmanBuilder(history_length = self.history_length, state_size = self.state_size, 
                                    num_actions = self.num_actions, double_q = self._double_q)
        inputs = {"discount": discount, "reward": reward, "terminal": terminal, "action": chosen,
                  "next_state": nstate, "state": state}
        b = bb.build(qbuilder = self._q_builder, value_scope = v.scope, 
                     target_scope = target_scope, inputs = inputs)
        self._summaries += b.summaries
        self._qnet.set_bellman_ops(b)

        with tf.name_scope("current_Q"):
            current_q     = choose_from_array(v.q_values, chosen)
            self._summaries.append(tf.summary.scalar("mean_Q", tf.reduce_mean(current_q)))

        self._build_training(optimizer, current_q, b.updated_q)
        self._qnet.set_summaries(tf.summary.merge(self._summaries))
        self._summaries = []

        return self._qnet


    def _build_training(self, optimizer, current_q, target_q):
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
