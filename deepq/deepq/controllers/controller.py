import tensorflow as tf
from ..action_space import ActionSpace

class Controller(object):
    def __init__(self, action_space):
        self._session           = None
        self._summary_writer    = None
        self._action_space      = ActionSpace(action_space)

        # tracking variables
        self._last_action = None
        self._frame_count = 0

    @property
    def session(self):
        return self._session

    @property
    def action_space(self):
        return self._action_space

    @property
    def frame_count(self):
        return self._frame_count

    def observe(self, state, reward, test=False):
        if not test:
            self._frame_count += 1
        self._observe(state = state, reward = reward, test = test)

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