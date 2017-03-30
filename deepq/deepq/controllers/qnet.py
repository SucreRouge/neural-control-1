import numpy as np
import tensorflow as tf

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
        return session.run([self._value_out], feed_dict={self._value_in:state})[0]

    def _train_feed(self, qs):
        actions = np.reshape(qs.action, (len(qs.action),))
        feed = {self._train_current:  qs.current,
                self._train_next:     qs.next,
                self._train_chosen:   actions,
                self._train_reward:   qs.reward,
                self._train_terminal: qs.terminal}
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

class QNet(object):
    def __init__(self, state_size, history_length, num_actions, graph = None, 
                 target_net = True, double_q=False, dueling=False):
        if graph is None:
            graph = tf.get_default_graph()

        self._state_size     = state_size
        self._history_length = history_length
        self._num_actions    = num_actions
        self._graph          = graph
        self._double_q       = double_q
        self._use_target_net = target_net
        self._dueling_arch   = dueling
        self._summaries      = []

    def build_graph(self, arch, optimizer):
        with self._graph.as_default():
            self._summaries = []
            gstep = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
            self._qnet  = QNetGraph(gstep)

            self._build_value_network(arch)
            self._build_target_network(arch)

            self._qnet  = self._build_graph(arch, optimizer)
            self._qnet.set_summaries(tf.summary.merge(self._summaries))
            self._summaries = []

        return self._qnet

    def _build_value_network(self, arch):
        # build the value network
        with tf.variable_scope("value_network") as value_net_scope:
            value_input   = self._make_input()
            value_actions = self._build_q_net(value_input, arch)
            self._summaries.append(tf.summary.histogram("action_scores", value_actions))

        self._qnet.set_value_network(input = value_input, output = value_actions, scope = value_net_scope)

    def _build_target_network(self, arch):
        """ This function builds the target network that is used to generate the Q values for the
            next state in the bellman update.
            It propagates the next state through the network architecture. If '_use_target_net'
            is set to False, is reuses the weights of the value network.
        """
        assert self._qnet._value_scope is not None, "Cannot build target network when value network does not exist"
        # the target net
        if self._use_target_net:
            with tf.variable_scope("target_network") as target_scope:
                state = self._make_input()
                qvals = self._build_q_net(state, arch)
                self._summaries.append(tf.summary.histogram("action_scores", qvals))
        else:
            with tf.name_scope("target_network"):
                with tf.variable_scope(self._qnet._value_scope, reuse=True):
                    state = self._make_input()
                    qvals = self._build_q_net(state, arch)
            self._summaries.append(tf.summary.histogram("action_scores", qvals))

        if self._use_target_net:
            update_target = assign_from_scope(self._qnet._value_scope, target_scope, name="update_target_network")
        else:
            update_target = tf.no_op()

        self._qnet.set_target_network(input = state, output = qvals, update = update_target)


    def _build_graph(self, arch, optimizer):
        with tf.variable_scope("bellman_update"):
            reward   = tf.placeholder(tf.float32, [None], name="reward")
            chosen   = tf.placeholder(tf.int32,   [None], name="chosen")
            terminal = tf.placeholder(tf.bool,    [None], name="terminal")
            discount = tf.Variable(0.99, dtype=tf.float32, trainable=False, name='discount')

            target_input   = self._qnet._target_in
            target_actions = self._qnet._target_out
            value_input    = self._qnet._value_in
            value_actions  = self._qnet._value_out


            # TODO debug to check that this does what i think it does
            # target vaues
            with tf.name_scope("future_reward"):
                if self._double_q:
                    with tf.name_scope("double_q"):
                        with tf.variable_scope(self._qnet._value_scope, reuse=True):
                            proposed_actions = self._build_q_net(target_input, arch)
                        best_action = tf.argmax(proposed_actions, axis=1, name="best_action")
                    best_future_q = choose_from_array(target_actions, best_action)
                else:
                    best_future_q = tf.reduce_max(target_actions, axis=1, name="best_future_Q")                
                future_rwd = best_future_q * (1.0 - tf.to_float(terminal))

            with tf.name_scope("updated_Q"):
                state_value = discount * future_rwd + reward

            # current values
            with tf.name_scope("current_Q"):
                current_q     = choose_from_array(value_actions, chosen)
                self._summaries.append(tf.summary.scalar("mean_Q", tf.reduce_mean(current_q)))


        with tf.variable_scope("training"):
            num_samples = tf.shape(current_q)[0]
            loss    = tf.losses.mean_squared_error(current_q, tf.stop_gradient(state_value), scope='loss')
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
        
        self._qnet.set_training_ops(current = value_input, next = target_input, chosen = chosen, reward = reward,
                            terminal = terminal, loss = loss, train = train)
        return self._qnet


    def _make_input(self, name="state"):
        return tf.placeholder(tf.float32, [None, self._history_length, self._state_size], name=name)


    def _build_q_net(self, input, arch):
        history_length = input.get_shape()[1].value
        state_size     = input.get_shape()[2].value
        num_actions    = self._num_actions
        features = arch(input)
        if self._dueling_arch:
            return self._make_dueling_arch(features, num_actions)
        else:
            actions  = tf.layers.dense(features, num_actions, name="qvalues")
            return actions

    def _make_dueling_arch(self, features, num_actions):
        action_vals = tf.layers.dense(features, num_actions, name="action_values")
        state_vals  = tf.layers.dense(features, 1, name="state_values")
        with tf.name_scope("q_values"):
            q_vals = state_vals + action_vals - tf.reduce_mean(action_vals, axis=1, keep_dims=True)
        return q_vals  

# tf helper functions
def assign_from_scope(source_scope, target_scope, name=None):
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name
    if isinstance(target_scope, tf.VariableScope):
        target_scope = target_scope.name

    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    with tf.name_scope(name):
        for source in source_vars:
            for target in target_vars:
                source_name = source.name[len(source_scope):]
                target_name = target.name[len(target_scope):]
                if source_name == target_name:
                    asgns.append(target.assign(source))
        return tf.group(*asgns)

def choose_from_array(source, indices):
    """ returns [source[i, indices[i]] for i in 1:len(indices)] """
    with tf.name_scope("choose_from_array"):
        num_samples = tf.shape(indices)[0]
        indices     = tf.transpose(tf.stack([tf.cast(tf.range(0, num_samples), indices.dtype), indices]))
        values      = tf.gather_nd(source, indices)
    return values

