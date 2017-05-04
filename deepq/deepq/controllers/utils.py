import tensorflow as tf
import collections

def current_name_scope():
    return tf.get_default_graph()._name_stack + "/"

# tf helper functions
def assign_from_scope(source_scope, target_scope, name=None):
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name
    if isinstance(target_scope, tf.VariableScope):
        target_scope = target_scope.name

    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    with tf.name_scope(name):
        for source in source_vars:
            for target in target_vars:
                source_name = source.name[len(source_scope):]
                target_name = target.name[len(target_scope):]
                if source_name == target_name:
                    asgns.append(target.assign(source))
        return tf.group(*asgns)

def update_from_scope(source_scope, target_scope, rate, name="soft_update"):
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name
    if isinstance(target_scope, tf.VariableScope):
        target_scope = target_scope.name

    source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)
    asgns = []
    with tf.name_scope(name):
        for source in source_vars:
            for target in target_vars:
                source_name = source.name[len(source_scope)+1:]
                target_name = target.name[len(target_scope)+1:]
                if source_name == target_name:
                    sn = source_name.split(":")[0]
                    with tf.name_scope(sn):
                        newval = (1 - rate) * target + rate * source
                        asgns.append(target.assign(newval))
        return tf.group(*asgns)

def choose_from_array(source, indices, name="choose_from_array"):
    """ returns [source[i, indices[i]] for i in 1:len(indices)] """
    with tf.name_scope(name):
        num_samples = tf.shape(indices)[0]
        indices     = tf.transpose(tf.stack([tf.cast(tf.range(0, num_samples), indices.dtype), indices]))
        values      = tf.gather_nd(source, indices)
    return values

def copy_variables_to_scope(source_scope, target_scope, trainable=None):
    if isinstance(source_scope, tf.VariableScope):
        source_scope = source_scope.name

    sources = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)

    with tf.variable_scope(target_scope, reuse=False) as tscope:
        with tf.name_scope(tscope.name+"/"):
            for var in sources:
                source_name = var.name[len(source_scope):]
                if source_name.startswith("/"):
                    source_name = source_name[1:]
                source_name = source_name.split(":")[0]
                newvar = tf.get_variable(name = source_name, initializer = var.initialized_value(), trainable=trainable)
    return tscope

def summarize_gradients(gradients_and_vars, histograms = False):
    summaries = []
    for g, v in gradients_and_vars:
        with tf.name_scope(v.name.split(":")[0]):
            if histograms:
                summaries.append(tf.summary.histogram("histogram", g))
            
            # to get a sense of the size of the gradient, calculate its root mean square
            # (I think that makes more sense than using the L^2 norm, as that depends on 
            # the size of the variable)
            with tf.name_scope("calc_rms"):
                nsq = tf.reduce_mean(tf.square(g))
                rms = tf.sqrt(nsq, name="rms")
            summaries.append(tf.summary.scalar("rms", rms))
    return tf.summary.merge(summaries, name="gradient_summary")
        

def total_size(tensor):
    shape = tensor.get_shape()
    size = 1
    for s in shape:
        size *= s.value
    return size

def apply_binary_op(a, b, op, **kwargs):
    """ applies the binary operation op to a and b
        if either of them is None, the other is returned,
        if both are None, None is returned.

        If a and b are iterables, this applies op 
        elementwise and returns a list.
    """

    if isinstance(a, collections.Iterable):
        if not isinstance(b, collections.Iterable):
            b = repeat(b)
    elif isinstance(b, collections.Iterable):
        a = repeat(a)
    else:
        return _apply_binary_op(a, b, op, **kwargs)

    # if we get to here, we have to iterables
    return list(map(lambda x:_apply_binary_op(x[0], x[1], op, **kwargs), zip(a, b)))

def _apply_binary_op(a, b, op, **kwargs):
    if a is not None:
        if b is not None:
            return op(a, b, **kwargs)
        else:
            return a
    elif b is not None:
        return b
    else:
        return None