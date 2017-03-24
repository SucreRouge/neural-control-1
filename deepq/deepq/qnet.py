import numpy as np
import tensorflow as tf

class QNetGraph(object):
    def __init__(self, update):
        self._update_target = update

    def set_policy_ops(self, input, output):
        self._policy_in  = input
        self._policy_out = output

    def set_training_ops(self, current, chosen, reward, next, terminal, loss, train):
        self._train_current  = current
        self._train_chosen   = chosen 
        self._train_reward   = reward 
        self._train_next     = next
        self._train_terminal = terminal
        self._train_loss     = loss
        self._train_step     = train

    def get_actions(self, state, session):
        # batchify state
        state = state[np.newaxis, :]
        return session.run([self._policy_out], feed_dict={self._policy_in:state})[0]

    def _train_feed(self, qs):
        feed = {self._train_current:  qs.current,
                self._train_next:     qs.next,
                self._train_chosen:   qs.action,
                self._train_reward:   qs.reward,
                self._train_terminal: qs.terminal}
        return feed

    def train_step(self, qs, session):
        feed = self._train_feed(qs)
        _, loss = session.run([self._train_step, self._train_loss], feed_dict = feed)
        return loss

    def update_target(self, session):
        session.run(self._update_target)

class QNet(object):
    def __init__(self, state_size, history_length, num_actions, graph = None):
        if graph is None:
            graph = tf.get_default_graph()

        self._state_size     = state_size
        self._history_length = history_length
        self._num_actions    = num_actions
        self._graph          = graph

    def build_graph(self, arch, optimizer):
        with self._graph.as_default():
            self._qng = self._build_graph(arch, optimizer)

        return self._qng

    def _build_graph(self, arch, optimizer):
        # the target net
        with tf.variable_scope("target_network"):
            target_input, target_actions = build_q_net(self._state_size, self._history_length, self._num_actions, arch)

        with tf.variable_scope("value_network"):
            value_input, value_actions   = build_q_net(self._state_size, self._history_length, self._num_actions, arch)

        with tf.variable_scope("training"):
            reward   = tf.placeholder(tf.float32, [None], name="reward")
            chosen   = tf.placeholder(tf.int32, [None], name="chosen")
            terminal = tf.placeholder(tf.bool, [None], name="chosen")
            discount = tf.Variable(0.99, dtype=tf.float32, trainable=False)

            # TODO debug to check that this does what i think it does
            # target vaues
            with tf.name_scope("target_Q"):
                best_future_q = tf.reduce_max(target_actions, axis=1)
                state_value   = best_future_q * discount * tf.to_float(terminal) + reward

            # current values
            with tf.name_scope("current_Q"):
                current_q     = choose_from_array(value_actions, chosen)

            loss  = tf.losses.mean_squared_error(current_q, tf.stop_gradient(state_value))
            train = optimizer.minimize(loss)

        with tf.variable_scope("update_target_network"):
            update_target = assign_from_scope("value_network", "target_network")

        qng = QNetGraph(update = update_target)
        qng.set_policy_ops(input = value_input, output = value_actions)
        qng.set_training_ops(current = value_input, next = target_input, chosen = chosen, reward = reward,
                            terminal = terminal, loss = loss, train = train)
        return qng

# tf helper functions
def assign_from_scope(source_scope, target_scope):
    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    for source in source_vars:
        for target in target_vars:
            source_name = source.name[len(source_scope):]
            target_name = target.name[len(target_scope):]
            if source_name == target_name:
                asgns.append(target.assign(source))
    return tf.group(*asgns)

def choose_from_array(source, indices):
    """ returns [source[i, indices[i]] for i in 1:len(indices)] """
    num_samples = tf.shape(indices)[0]
    indices     = tf.transpose(tf.stack([tf.range(0, num_samples), indices]))
    values      = tf.gather_nd(source, indices)
    return values

def build_q_net(state_size, history_length, num_actions, arch):
    inp = tf.placeholder(tf.float32, [None, history_length, state_size], name="state")
    features = arch(inp)
    actions  = tf.layers.dense(features, num_actions, name="qvalues")

    return inp, tf.reshape(actions, [-1, num_actions])