from .networks import ActorCriticBuilder
from .controller import Controller
from ..action_space import ActionSpace, rescale
from .memory import History, Memory
import tensorflow as tf
import numpy as np

from deepq import noise

class ExplorationPolicy(object):
    def __init__(self, start_eps, end_eps, num_steps, noise):
        self._start_epsilon = start_eps
        self._end_epsilon   = end_eps
        self._num_steps     = num_steps
        self._noise         = noise
        self.epsilon = start_eps

    def set_stepcount(self, steps):
        decay = min(1, float(steps) / self._num_steps)
        dist  = self._start_epsilon - self._end_epsilon
        self.epsilon = self._start_epsilon - decay * dist

    def __call__(self, actions, test=False):
        if not test:
            a = np.random.rand(*actions.shape) * 2 - 1
            return actions + a * self.epsilon
        else:
            return actions

class DeepPolicyGradientController(Controller):
    def __init__(self, history_length, memory_size, state_size, action_space,
                    steps_per_epoch=10000, minibatch_size=64, final_exploration_frame=1000000,
                    explorative_noise=None, final_epsilon=0.1, warmup_time=10000, 
                    policy_warmup_time=20000, discount=0.99):
        action_space = ActionSpace(action_space)
        assert not action_space.is_discrete, "DeepPolicyGradientController works only on continuous action spaces"
        o = np.ones(action_space.num_actions)
        action_space = rescale(action_space, -o, o)
        super(DeepPolicyGradientController, self).__init__(action_space, state_size, history_length)

        if explorative_noise is None:
            explorative_noise = noise.GaussianWhiteNoise(np.random)

        self._num_actions     = action_space.num_actions[0]
        self._steps_per_epoch = steps_per_epoch
        self._warmup_time     = warmup_time
        self._policy_warmup_time = policy_warmup_time
        self._next_epoch      = None
        self._minibatch_size  = minibatch_size
        self._explore_policy  = ExplorationPolicy(1.0, final_epsilon, final_exploration_frame, explorative_noise)
        self._state_memory    = Memory(size=int(memory_size), history_length=history_length, state_size=state_size,
                                       action_dim = self._num_actions, action_type = float)

        # counters
        self._step_counter    = 0
        self._epoch_counter   = 0
        self._discount        = discount

    def _observe(self, state, last, reward, action, test=False):
        # if this is the first state, there is no transition to remember,
        # so simply add to the state history
        if action is None:
            return

        # if not in test mode, remember the transition
        if not test:
            self._state_memory.append(state=last, next=state, reward=reward, action=action)

    def _get_action(self, test=False):
        action_vals   = self._qnet.get_actions(self.full_state, self.session)
        action        = self._explore_policy(action_vals, test)
        action        = np.clip(action, -1, 1)
        if not test:
            self._explore_policy.set_stepcount(max(0, self.frame_count - self._warmup_time))
        
        return action, action_vals

    def train(self):
        if len(self._state_memory) < self._warmup_time:
            return

        summary_writer = self._summary_writer if self._step_counter % 100 == 0 else None

        sample = self._state_memory.sample(self._minibatch_size)

        # do a few steps of critic training before using the critic to train the policy.
        if self._epoch_counter * self._steps_per_epoch + self._step_counter < self._policy_warmup_time:
            ls, Q = self._qnet.train_step_critic(sample, self._session, summary_writer)
        else:
            ls, Q = self._qnet.train_step(sample, self._session, summary_writer)

        # break on divergent behaviour
        if np.mean(np.abs(Q)) > 1e6:
            raise Exception("Critic network seems to have diverged! Try different hyperparameters.")

        # in case of soft updates, we update after each train step
        if self._soft_target_update:
            self._qnet.update_target(self._session)

        self._step_counter += 1
        if self._step_counter > self._steps_per_epoch:
            # copy target net to policy net if we do hard target updates
            if not self._soft_target_update:
                self._qnet.update_target(self._session)
            
            smr, step = self.session.run([self._var_summaries, self._qnet._global_step])
            self._summary_writer.add_summary(smr, step)

            self._step_counter = 0
            self._epoch_counter += 1
            if self._next_epoch:
                self._next_epoch()

    def save(self, name):
        self._saver.save(self.session, name, global_step=self._qnet.global_step)

    def setup_graph(self, actor_net, critic_net, graph = None, actor_learning_rate = 1e-4, critic_learning_rate = 1e-3, 
                          soft_target = False, global_step = None, critic_regularizer = None,
                          critic_init = None, policy_init = None):
        qnet = ActorCriticBuilder(state_size     = self._state_size, 
                    history_length  = self.history_length, 
                    num_actions     = self._num_actions,
                    soft_target_update = soft_target,
                    critic_net      = critic_net,
                    policy_net      = actor_net,
                    discount        = self._discount,
                    critic_regularizer = critic_regularizer,
                    critic_init     = critic_init,
                    policy_init     = policy_init)

        self._soft_target_update = soft_target

        # TODO Figure these out!
        aopt = tf.train.AdamOptimizer(learning_rate=actor_learning_rate)
        copt = tf.train.AdamOptimizer(learning_rate=critic_learning_rate)
        self._qnet = qnet.build(actor_optimizer = aopt, critic_optimizer = copt, graph = graph, 
                                gstep = global_step)

        # setup variable statistics
        with tf.name_scope("weight_summaries"):
            summaries = [tf.summary.histogram(var.name, var) for var in tf.trainable_variables()]
            self._var_summaries = tf.summary.merge(summaries)

        # setup saver
        self._saver = tf.train.Saver()

    # these might help debugging/understanding stuff
    def get_Q(self, states, actions):
        return self._qnet.critic.critique(state = states, action = actions, session = self.session)
