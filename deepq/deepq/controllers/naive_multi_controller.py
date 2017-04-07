from .controller import Controller

class FakePolicy:
    def __init__(self):
        self.epsilon = 1.0

class NaiveMultiController(Controller):
    def __init__(self, sub_controllers, action_space, state_size):
        super(NaiveMultiController, self).__init__(action_space, state_size)

        self._sub_controllers = sub_controllers
        self._epoch_counter   = 0
        self._policy = FakePolicy()

    def observe(self, state, reward, test=False):
        # call super
        super(NaiveMultiController, self).observe(state=state, reward=reward, test=test)
        
        # and relay raw data to sub-controllers
        for ctrl in self._sub_controllers:
            ctrl.observe(state, reward, test)

    def _get_action(self, test=False):
        action_value_tuples = [ctrl.get_action(test) for ctrl in self._sub_controllers]
        actions = [a for (a, v) in action_value_tuples]
        values  = [v for (a, v) in action_value_tuples]
        return actions, values

    def init(self, session, logger):
        super(NaiveMultiController, self).init(session, logger)
        
        for ctrl in self._sub_controllers:
            ctrl.init(session, logger)

    def train(self):
        for ctrl in self._sub_controllers:
            ctrl.train()