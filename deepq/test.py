from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


# Code taken from: 
#  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

# Helper functions for TF Graph visualization

import tensorflow as tf
import numpy as np
  
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

import tensorflow as tf
import numpy as np
from deepq import *
from deepq.control_env import ControlEnv
from deepq.controllers.networks.discrete_q import *
from gym import spaces

graph = tf.Graph()
session = tf.Session()

def arch(inp):
    s = [d.value for d in inp.get_shape()]
    flat = tf.reshape(inp, [-1, s[1]*s[2]])
    fc = tf.layers.dense(flat, 128, name="fc")
    return fc

controller = DiscreteDeepQController(history_length=10, memory_size=1000000, 
              state_size=10, action_space=spaces.Discrete(5),
              final_exploration_frame=1e5, minibatch_size=32)
controller.setup_graph(arch, double_q=True, target_net=True, dueling=True, learning_rate=2.5e-4, graph=graph)

show_graph(graph)