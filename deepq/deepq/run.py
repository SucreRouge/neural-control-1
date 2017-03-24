import numpy as np
from collections import namedtuple, deque

run_result=namedtuple("RunResult", ["total_reward", "episode_length", "expected_reward", "mean_q"])

class ControlRun(object):
    def __init__(self, controller, task, test_every):
        self._controller = controller
        self._task       = task
        self._on_finish_episode = None
        self._on_test    = None

        self._next_test  = 0
        self._test_every = test_every
    
    def set_callbacks(self, episode, test):
        self._on_finish_episode = episode
        self._on_test = test

    def run(self, num_frames):
        total_frames = 0
        while total_frames < num_frames:
            result = self.run_episode(test=False)
            total_frames += result.episode_length
            if self._on_finish_episode:
                self._on_finish_episode(result)
            self._task.reset()
            if total_frames > self._next_test:
                self.test()
                self._next_test += self._test_every
                
    def run_episode(self, test=False, record=False):
        # shortcuts
        task       = self._task
        controller = self._controller 
        
        # trackers
        total_reward    = 0.0
        total_q         = 0.0
        expected_reward = 0.0
        episode_length  = 0
        if record:
            track = deque()
        while True:
            action, values   = controller.get_action(test=test)
            reward, terminal = task.update(action)
            q                = np.amax(values)
            total_q         += q
            expected_reward += total_reward + q
            total_reward    += reward
            episode_length  += 1
            controller.observe(state=None if terminal else task.state, reward=reward, test=test)
            
            if record:
                track.append(np.array([task._system.x, task._control]))

            if terminal:
                result = run_result(total_reward, episode_length, expected_reward / episode_length, 
                                  total_q / episode_length)
                if record:
                    return result, np.array(track)
                else:
                    return result

            if not test:
                self._controller.train()

    def test(self):
        # start a new test task here
        self._task.reset(test=True)
        result, track = self.run_episode(test=True, record=True)
        self._on_test(result, track)
        self._task.reset(test=False)

def run(controller, task, num_frames, test_every, episode_callback=None, test_callback=None):
    r = ControlRun(controller, task, test_every)
    r.set_callbacks(episode = episode_callback, test = test_callback)
    r.run(num_frames)