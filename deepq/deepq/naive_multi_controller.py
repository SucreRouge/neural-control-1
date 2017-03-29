from .controller import Controller

class FakePolicy:
    def __init__(self):
        self.epsilon = 1.0

class NaiveMultiController(Controller):
    def __init__(self, sub_controllers, action_space):
        super(NaiveMultiController, self).__init__(action_space)

        self._sub_controllers = sub_controllers
        self._action_space    = action_space
        self._epoch_counter   = 0
        self._policy = FakePolicy()
        self._action_counter  = 0

    def observe(self, state, reward, test=False):
        for ctrl in self._sub_controllers:
            ctrl.observe(state, reward, test)

    def get_action(self, test=False):
        action_value_tuples = [ctrl.get_action(test) for ctrl in self._sub_controllers]
        actions = self._action_space.get_action([a for (a, v) in action_value_tuples])
        values  = [v for (a, v) in action_value_tuples]
        self._action_counter += 1
        return actions, values

    def init(self, session, logger):
        super(NaiveMultiController, self).init(session, logger)
        
        for ctrl in self._sub_controllers:
            ctrl.init(session, logger)

    def train(self):
        for ctrl in self._sub_controllers:
            ctrl.train()