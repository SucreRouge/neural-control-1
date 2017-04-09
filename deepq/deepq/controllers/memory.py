from collections import deque, namedtuple
import numpy as np

def check_np_array(obj, dims=None):
  if not isinstance(obj, np.ndarray):
    raise TypeError("Expect numpy array")
  if dims is not None and len(obj.shape) != dims:
    raise ValueError("Got %d dimensional array, expected %d dimensions"%(len(object.shape), dims))
  return True


class History(object):
    """ This class is responsible for tracking the history of past states.
    """

    def __init__(self, duration, state_size):
        self._duration   = duration
        self._state_size = state_size
        self._history = deque(maxlen = duration)
        # fill with zeros
        self.clear()

    @property
    def duration(self):
        return self._duration

    @property
    def state_size(self):
        return self._state_size

    @property
    def state(self):
        return np.array(self._history)

    def observe(self, state):
        check_np_array(state, 1)
        assert state.shape[0] == self.state_size, "%s != %s"%(state.shape[0], self.state_size)

        self._history.append(state)
        return self.state

    def clear(self):
        for i in range(self.duration):
            self._history.append(np.zeros(self.state_size))


QSample = namedtuple('QSample', ['current', 'action', "reward", "next", "terminal"])

class Memory(object):
    """ This class is responsible for managing the replay memory
    """
    def __init__(self, size, history_length, state_size, action_dim = 1, action_type = np.int32):
        """ size: int Number of states transitions to remember
            history_length: int Number of frames that are save in history.
            state_size: int Size of a single state. 
            action_dim: int Number of independent actions.
            A complete state history thus has the shape (history_length, state_size)
        """
        self._size    = size
        self._states  = np.zeros((size, history_length, state_size))
        self._actions = np.zeros((size, action_dim), dtype=action_type)
        self._rewards = np.zeros(size)
        self._next    = np.zeros((size, history_length, state_size))
        self._term    = np.zeros(size, dtype=bool)
        self._write_pointer = 0
        self._total   = 0

    def append(self, state, action, reward, next):
        i = self._write_pointer
        self._states[i, :, :] = state
        self._actions[i, :]   = action
        self._rewards[i]      = reward
        self._term[i]         = next is None
        if not self._term[i]:
            self._next[i, :, :] = next

        # advance pointer
        self._write_pointer += 1
        if self._write_pointer == self._size:
            self._write_pointer = 0
        self._total += 1

    def sample(self, amount):
        s  = np.random.randint(len(self), size=amount)
        return QSample(self._states[s], self._actions[s], self._rewards[s], self._next[s], self._term[s])

    def __len__(self):
        return min(self._size, self._total)