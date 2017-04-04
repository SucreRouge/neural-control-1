import tensorflow as tf
from ..action_space import ActionSpace
from .memory import History

class Controller(object):
    def __init__(self, action_space, state_size, history_length = 1):
        self._session           = None
        self._summary_writer    = None
        self._action_space      = ActionSpace(action_space)

        #TODO use state_space instead of state_size
        self._state_size     = state_size
        self._history_length = history_length
        self._history        = History(duration=history_length, state_size=state_size)

        # tracking variables
        self._last_action = None
        self._frame_count = 0

    # getter properties
    @property
    def session(self):
        return self._session

    @property
    def action_space(self):
        return self._action_space

    @property
    def state_size(self):
        return self._state_size

    @property
    def history_length(self):
        return self._history_length

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def full_state(self):
        return self._history.state



    def observe(self, state, reward, test=False):
        if not test:
            self._frame_count += 1

        action = self._last_action
        #TODO we can cache this
        last_state = self.full_state

        if state is not None:
            state = self._history.observe(state)
        else:
            state = None
            self._last_action = None
            self._history.clear()
        
        self._observe(state = state, last = last_state, reward = reward, action = action, test = test)

    def get_action(self, test = False):
        a, v = self._get_action(test)
        self._last_action = a
        return self.action_space.get_action(a), v

    def _get_action(self, test):
        raise NotImplementedError()

    def _observe(self, state, reward, test):
        raise NotImplementedError()

    def train(self):
        pass

    def init(self, session = None, logger = None):
        if session is not None and not isinstance(session, tf.Session):
            raise TypeError("Expected a tensorflow Session or None, got %r"%session)

        self._session        = session
        self._summary_writer = logger
        # this might be a bit wasteful! I don't know.
        self._session.run([tf.global_variables_initializer()])