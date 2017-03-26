"""
This file is responsible for testing that the tensorflow graphs built by the QNet class
actually do what they are supposed to.
"""

from os import path
import sys
sys.path.append( path.dirname(path.dirname( path.abspath(__file__) )  ))

from qnet import QNet, QNetGraph
import tensorflow as tf
import numpy as np

counter = np.array([0])

def mini_arch(inp):
    global counter
    v = tf.get_variable(name="MARKER", shape=(1,), initializer=tf.constant_initializer(counter))
    counter += 1
    return inp

def count_markers():
    var = tf.global_variables()
    var = [v for v in var if v.name.endswith("MARKER:0")]
    return len(var)

def test_target_net(use_target, **kwargs):
    g = tf.Graph()
    net = QNet(1, 1, 1, graph = g, target_net = use_target, **kwargs)
    sess = tf.Session(graph=g)
    with g.as_default(), sess:
        gstep = tf.Variable(0,    dtype=tf.int64,   trainable=False, name="global_step")
        net._qnet  = QNetGraph(gstep)
        net._build_value_network(mini_arch)

        # at this point, g contains one MARKER variable
        # we can verify that:
        assert count_markers() == 1, "Expect one MARKER variable, something is wrong with the test"

        # Now, depending on whether we want seperate vars for target and value, we expect one or two counters
        net._build_target_network(mini_arch)
        if use_target:
            assert count_markers() == 2, "Expect two MARKER variable, target network was not correctly built."
        else:
            assert count_markers() == 1, "Expect one MARKER variable, target network seems to reuse wrongly."

        if use_target:
            sess.run(tf.global_variables_initializer())
            gv = tf.GraphKeys.GLOBAL_VARIABLES
            target_marker = [v for v in tf.get_collection(gv, scope="target_network") if v.name.endswith("MARKER:0")]
            old_value_target = target_marker[0].eval()
            value_marker  = [v for v in tf.get_collection(gv, scope="value_network") if v.name.endswith("MARKER:0")]
            old_value_source = value_marker[0].eval()

            assert old_value_source != old_value_target, "Target and Value nets are initialized differently"
            net._qnet.update_target(session=sess)

            assert value_marker[0].eval() == old_value_source, "Update should not change value net"
            assert target_marker[0].eval() == old_value_source, "Update should set target net to value net"


test_target_net(True, dueling=False)
test_target_net(True, dueling=True)
test_target_net(False, dueling=False)
test_target_net(False, dueling=True)
