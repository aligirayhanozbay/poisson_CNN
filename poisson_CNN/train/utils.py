import tensorflow as tf

def choose_optimizer(name):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam
    elif name == "sgd":
        return tf.keras.optimizers.SGD

def load_model_checkpoint(model, checkpoint_path, model_config = None, sample_model_input = None):
    if checkpoint_path is not None:
        checkpoint_filename = tf.train.latest_checkpoint(checkpoint_path)
        print('Attempting to load checkpoint from ' + checkpoint_filename)
        try:
            model.load_weights(checkpoint_filename)
        except:#tf is stupid and is incapable of casting weights saved as eg float32 to float64. handle that problem.
            original_floatx = tf.keras.backend.floatx()
            checkpoint = tf.train.load_checkpoint(checkpoint_filename)
            weight_name_list = tf.train.list_variables(checkpoint_filename)
            checkpoint_dtype = checkpoint.get_tensor(weight_name_list[1][0]).dtype
            tf.keras.backend.set_floatx(str(checkpoint_dtype))
            dummy_model = type(model)(**model_config)
            _ = dummy_model([tf.cast(t, tf.keras.backend.floatx()) for t in sample_model_input])
            dummy_model.load_weights(checkpoint_filename)
            tf.keras.backend.set_floatx(original_floatx)
            model.set_weights(dummy_model.get_weights())
            del dummy_model, checkpoint, weight_name_list, checkpoint_dtype
    else:
        pass

