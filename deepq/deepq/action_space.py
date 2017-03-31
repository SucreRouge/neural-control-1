"""
    Utilities to better work with 
"""

from gym import spaces
import numpy as np
import itertools
import numbers

def is_discrete(space):
    # check if space is discrete
    if isinstance(space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, spaces.Box):
        return False
    elif isinstance(space, GenericActionSpace):
        return space.is_discrete
    raise TypeError("Unknown space %r supplied"%type(space))

def is_compound(space):
    if isinstance(space, spaces.Discrete):
        return False
    elif isinstance(space, spaces.Box):
        return len(space.shape) != 1 or space.shape[0] != 1
    elif isinstance(space, (spaces.MultiDiscrete, spaces.MultiBinary)):
        return True
    elif isinstance(space, GenericActionSpace):
        return space.is_compound

    raise TypeError("Unknown space %r supplied"%type(space))

class GenericActionSpace(object):
    def __init__(self, space):
        self._space = space
        self._discrete = is_discrete(space)
        self._compound = is_compound(space)
        
        # TODO tuple is not so trivial...
        if isinstance(space, spaces.Discrete):
            self._num_actions = space.n
        elif isinstance(space, spaces.Box):
            self._num_actions = space.shape

    @property
    def is_discrete(self):
        return self._discrete

    @property
    def is_compound(self):
        return self._compound

    # generic interface so different algorithms can perform control
    @property
    def num_actions(self):
        """ Get the available number of actions. Makes sense only for 
            discrete action spaces 
        """
        return self._num_actions

    def get_action(self, x):
        """ Convert an action into something the environment will understand. 
            This is important in case some wrappers where applied.
        """
        return x

    def sample(self):
        return self._space.sample()

    def contains(self, x):
        return self._space.contains(x)

    def flattened(self):
        return flatten(self)

    def discretized(self, steps):
        return discretize(self, steps)

    def __repr__(self):
        return "ActionSpace(%r)"%(self._space)

def ActionSpace(space):
    if isinstance(space, GenericActionSpace):
        return space 
    else:
        return GenericActionSpace(space)

class WrappedActionSpace(GenericActionSpace):
    def __init__(self, space, wrapped, converter):
        super(WrappedActionSpace, self).__init__(space)
        self._wrapped   = wrapped
        self._converter = converter
    
    def get_action(self, x):
        return self._wrapped.get_action(self._converter(x))

    def __repr__(self):
        return "WrappedActionSpace(%r, %r, %r)"%(self._space, self._wrapped, self._converter)


# Discretization 
def discretize(space, steps):
    # leave an already discrete space unchanged
    if is_discrete(space):
        return space

    # unwrap ActionSpace instances
    if isinstance(space, GenericActionSpace):
        ds, cv = _discretize(space._space, steps)
        return WrappedActionSpace(ds, space, cv)

    ds, cv = _discretize(space, steps)
    return WrappedActionSpace(ds, GenericActionSpace(space), cv)

def _discretize(space, steps):
    if isinstance(space, spaces.Box):
        if len(space.shape) == 1 and space.shape[0] == 1:
            discrete_space = spaces.Discrete(steps)
            lo = space.low[0]
            hi = space.high[0]
            def convert(x):
                return lo + (hi - lo) * float(x) / steps
            return discrete_space, convert
        else:
            if isinstance(steps, numbers.Integral):
                steps = np.full(space.low.shape, steps)
            assert steps.shape == space.shape, "supplied steps have invalid shape"
            starts = np.zeros_like(steps)
            discrete_space = spaces.MultiDiscrete(zip(starts.flatten(), steps.flatten()))
            lo = space.low.flatten()
            hi = space.high.flatten()
            def convert(x):
                return np.reshape(lo + (hi - lo) * x / steps, space.shape)
            return discrete_space, convert
    raise ValueError()

# Flattening
def flatten(space):
    # no need to do anything if already flat
    if not is_compound(space):
        return space

    # can't flatten continuous compound spaces
    if not is_discrete(space):
        raise ValueError("Trying to flatten a continuous space %r"%space)

    # unwrap ActionSpace instances
    if isinstance(space, GenericActionSpace):
        ds, cv = _flatten(space._space)
        return WrappedActionSpace(ds, space, cv)

    ds, cv = _flatten(space)
    return WrappedActionSpace(ds, GenericActionSpace(space), cv)

def _flatten(space):
    if isinstance(space, spaces.MultiDiscrete):
        ranges = [range(space.low[i], space.high[i], 1) for i in range(space.num_discrete_space)]
        prod   = itertools.product(*ranges)
        lookup = list(prod)
        dspace = spaces.Discrete(len(lookup))
        convert = lambda x: lookup[x]
        return dspace, convert

    raise TypeError("Cannot flatten %r"%type(space))
