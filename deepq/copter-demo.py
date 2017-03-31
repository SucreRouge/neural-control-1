import tensorflow as tf
import numpy as np
from collections import deque
import time
import math
from deepq import *
from deepq.copter_env import CopterEnv

import gym
dt = 0.1

################################################################################
#                             Testing Stuff
################################################################################

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

def episode_callback():
    reward_hist   = deque()
    expected_hist = deque()
    frame_hist    = deque()
    def call(frame, result):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        frame_hist.append(frame)
        i = len(reward_hist)
        if i % 100 == 0:
            rwd_h = np.array(reward_hist)
            exp_h = np.array(expected_hist)
            frm_h = np.array(frame_hist)
            c     = np.stack([frm_h, rwd_h, exp_h]).transpose()
            np.savetxt("progress.txt", c)
            if len(reward_hist) > 100:
                print(np.mean(np.array(reward_hist)[-100:]))
    return call

def test_callback():
    reward_hist   = deque()
    expected_hist = deque()
    q_hist        = deque()
    def call(result, track):
        reward_hist.append(result.total_reward)
        expected_hist.append(result.expected_reward)
        q_hist.append(result.mean_q)
        epoch   = controller._epoch_counter
        epsilon = 0#controller._policy.epsilon
        acount  = controller.frame_count
        test_counter = len(reward_hist)

        # plot the test run
        fig, ax = plt.subplots(1,1)
        ax.set_title("Epoch: %d , Epsilon=%.1f%%, Score=%.2f"%(epoch, epsilon*100, result.total_reward))
        ax.set_autoscaley_on(False)
        ax.set_ylim([-25, 25])
        """
        ax.plot(track[:, 4])               # y
        ax.plot(track[:, 5])               # c
        ax.plot(track[:, 3])               # x
        """
        ax.plot(track[:, 6] * 180/math.pi)              # y
        ax.plot(track[:, 7] * 180/math.pi)              # c
        ax.plot(track[:, 8] * 180/math.pi)              # x
        fig.savefig("tests/test_%d.pdf"%test_counter)
        plt.close(fig)

        rwd_h = np.array(reward_hist)
        exp_h = np.array(expected_hist)
        q_h = np.array(q_hist)
        c     = np.stack([rwd_h, exp_h, q_h]).transpose()
        np.savetxt("testing.txt", c)


    return call

def arch(inp):
    c1 = tf.layers.conv1d(inp, 64, 3, padding='same', activation=tf.nn.relu, name="conv1")
    c2 = tf.layers.conv1d(c1,  32, 3, padding='same', activation=tf.nn.relu, name="conv2")

    s = [d.value for d in c2.get_shape()]
    flat = tf.reshape(c2, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 256, activation=tf.nn.relu, name="fc")
    return fc

task = CopterEnv()
use_single = False
use_cont   = True

# single controller
if use_cont:
    controller = DeepPolicyGradientController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=task.action_space,
              minibatch_size=32)
    def ff(input):
        return tf.layers.dense(input, 256, activation=tf.nn.relu, name="full_features")

    controller.setup_graph(arch, ff, target_net=True, learning_rate=2.5e-4)
elif use_single:
    controller = DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=ActionSpace(task.action_space).discretized(3),
              final_exploration_frame=2e5, minibatch_size=32)
    controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=2.5e-4)
# factored controller
else:
    controllers = [DiscreteDeepQController(history_length=10, memory_size=1e6, 
              state_size=task.observation_space.shape[0], action_space=action_space.spaces.Discrete(9),
              steps_per_epoch=20000, final_exploration_frame=5e5, minibatch_size=32) 
                  for i in range(4)]
    for (i, controller) in enumerate(controllers):
        with tf.variable_scope("controller_%d"%i):
            controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=1.0e-4)
    controller = NaiveMultiController(controllers, ActionSpace(task.action_space).discretized(9))

sw = tf.summary.FileWriter('./logs/', graph=tf.get_default_graph(), flush_secs=30)
controller.init(session=tf.Session(), logger=sw)

def show_graph(graph_def, max_const_size=32):
    import IPython.display as idisplay
    from IPython.display import clear_output, Image, display, HTML
    
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:800px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(graph_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:100%;height:100%;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    f = open("test.html", "w")
    f.write(HTML(iframe).data)
    f.close()

    display(HTML(iframe))

show_graph(tf.get_default_graph())

run(task=task, controller=controller, num_frames=1e7, test_every=2e4, 
    episode_callback=episode_callback(), test_callback = test_callback())