import numpy as np
from collections import namedtuple, deque
import numbers

run_result=namedtuple("RunResult", ["total_reward", "episode_length", "expected_reward", "mean_q"])

class ControlRun(object):
    def __init__(self, controller, task, test_every):
        self._controller = controller
        self._task       = task
        self._on_finish_episode = None
        self._on_test       = None
        self._on_test_step  = None
        self._on_train_step = None
        self._episode_count = 0

        self._next_test  = 0
        self._test_every = test_every

        self.reset_task()

    def reset_task(self):
        self._last_state = self._task.reset()
        return self._last_state
    
    def set_callbacks(self, episode, test, test_step = None, train_step = None):
        self._on_finish_episode = episode
        self._on_test = test
        self._on_test_step = test_step
        self._on_train_step = train_step

    def run(self, num_frames):
        total_frames = 0
        while total_frames < num_frames:
            onstep = self._on_train_step  if self._episode_count%100 == 0 else None 
            result = self.run_episode(test=False, onstep=onstep)
            total_frames += result.episode_length
            self._episode_count += 1
            if self._on_finish_episode:
                self._on_finish_episode(total_frames, result)
            self.reset_task()
            if total_frames > self._next_test:
                self.test()
                self._next_test += self._test_every
                
    def run_episode(self, test=False, record=False, onstep=None):
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

        # init with state given by reset_task
        controller.observe(self._last_state, reward=0.0, test=test)
        while True:
            action, values   = controller.get_action(test=test)
            observation, reward, terminal, info = task.step(action)
            self._last_state = observation
            q                = np.amax(values)
            total_q         += q
            expected_reward += total_reward + q
            total_reward    += reward
            episode_length  += 1
            controller.observe(state=None if terminal else observation, reward=reward, test=test)
            
            if record:
                if isinstance(action, numbers.Real):
                    track.append(np.concatenate([[episode_length, action, reward], observation]))
                else:
                    track.append(np.concatenate([[episode_length], action, [reward], observation]))
            if onstep:
                onstep(task)

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
        self.reset_task()
        result, track = self.run_episode(test=True, record=True, onstep=self._on_test_step)
        if self._on_test is not None:
            self._on_test(result, track)
        self.reset_task()

def run(controller, task, num_frames, test_every, episode_callback=None, test_callback=None, 
        test_step_callback=None, train_step_callback=None):
    r = ControlRun(controller, task, test_every)
    r.set_callbacks(episode = episode_callback, test = test_callback, test_step = test_step_callback,
                    train_step = train_step_callback)
    r.run(num_frames)