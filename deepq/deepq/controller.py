import tensorflow as tf
from .action_space import ActionSpace

class Controller(object):
    def __init__(self, action_space):
        self._session        = None
        self._summary_writer = None
        self._action_space   = ActionSpace(action_space)

    @property
    def session(self):
        return self._session

    @property
    def action_space(self):
        return self._action_space

    def observe(self, state, reward, test=False):
        pass

    def get_action(self, test=False):
        raise NotImplementedError()

    def train(self):
        pass

    def init(self, session = None, logger = None):
        if session is not None and not isinstance(session, tf.Session):
            raise TypeError("Expected a tensorflow Session or None, got %r"%session)

        self._session        = session
        self._summary_writer = logger