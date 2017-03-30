
class DeepPolicyGradientController(Controller):
    def __init__(self, history_length, memory_size, state_size, action_space,
                    steps_per_epoch=10000, minibatch_size=64):
        action_space = ActionSpace(action_space)
        assert not action_space.is_discrete, "DeepPolicyGradientController works only on continuous action spaces"
        super(DeepPolicyGradientController, self).__init__(action_space)

        self._num_actions     = action_space.num_actions
        self._state_size      = state_size
        self._history_length  = history_length
        self._steps_per_epoch = steps_per_epoch
        self._next_epoch      = None
        self._minibatch_size  = minibatch_size

        self._history         = History(duration=history_length, state_size=state_size)
        self._state_memory    = Memory(size=int(memory_size), history_length=history_length, state_size=state_size)

        # counters
        self._step_counter    = 0
        self._epoch_counter   = 0