import tensorflow as tf

def convert_tf_object_names(x):
    '''
    Converts tf object names in a list/dict and all member lists/dicts to actual Python objects. E.g. put {"activations":["tf.keras.leaky_relu"]} in a JSON, load it with the python json package and call this function to get a dict with a single key called "activations" with a list including the corresponding tensorflow function in it.
    
    Inputs:
    -x: List or dict.

    Output:
    -converted_x: A list or dict.
    '''
    if isinstance(x,list):
        converted_x = [eval(item) if (isinstance(item,str) and ('tf.' in item)) else convert_tf_object_names(item) if (isinstance(item, list) or isinstance(item, dict)) else item for item in x]
    elif isinstance(x,dict):
        converted_x = {key:(
            eval(x[key]) if (isinstance(x[key],str) and ('tf.' in x[key])) else convert_tf_object_names(x[key]) if (isinstance(x[key], list) or isinstance(x[key], dict)) else x[key])
                       for key in x}
    else:
        raise(ValueError('The input must be a list or dict'))
    return converted_x
