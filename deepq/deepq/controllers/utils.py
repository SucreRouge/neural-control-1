import tensorflow as tf

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

def copy_variables_to_scope(source_scope, target_scope):
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
                newvar = tf.get_variable(name = source_name, initializer = var.initialized_value())
    return tscope