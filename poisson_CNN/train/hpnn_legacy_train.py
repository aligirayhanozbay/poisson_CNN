import argparse, json
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from ..models import Homogeneous_Poisson_NN_Legacy
from ..losses import loss_wrapper
from ..dataset.generators import numerical_dataset_generator

def choose_optimizer(name):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam
    elif name == "sgd":
        return tf.keras.optimizers.SGD

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided", default=None)

args = parser.parse_args()

config = json.load(open(args.config))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

for key in config['model'].keys():
    if 'config' in key:
        for layer_config_key in config['model'][key].keys():
            if 'activation' in layer_config_key and isinstance(config['model'][key][layer_config_key],str):
                config['model'][key][layer_config_key] = eval(config['model'][key][layer_config_key])
                
model = Homogeneous_Poisson_NN_Legacy(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(**config['training']['loss_parameters'])
dataset = numerical_dataset_generator(**config['dataset'])

inp,tar=dataset.__getitem__(0)
out = model([inp[0][:1],inp[1][:1]])
model.compile(loss=loss,optimizer=optimizer)
cb = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt.checkpoint',save_weights_only=True,save_best_only=True,monitor = 'loss'),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
    tf.keras.callbacks.TerminateOnNaN()
]

if args.continue_from_checkpoint is not None:
    checkpoint_filename = tf.train.latest_checkpoint(args.continue_from_checkpoint)
    print(checkpoint_filename)
    try:
        model.load_weights(checkpoint_filename)
    except:#tf is stupid and is incapable of casting weights saved as eg float32 to float64. handle that problem.
        checkpoint = tf.train.load_checkpoint(checkpoint_filename)
        weight_name_list = tf.train.list_variables(checkpoint_filename)
        checkpoint_dtype = checkpoint.get_tensor(weight_name_list[1][0]).dtype
        tf.keras.backend.set_floatx(str(checkpoint_dtype))
        dummy_model = type(model)(**config['model'])
        _ = dummy_model([tf.cast(t, tf.keras.backend.floatx()) for t in inp])
        dummy_model.load_weights(checkpoint_filename)
        tf.keras.backend.set_floatx(config['training']['precision'])
        model.set_weights(dummy_model.get_weights())
        del dummy_model, checkpoint, weight_name_list, checkpoint_dtype


model.summary()
#model.run_eagerly = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model.fit(dataset,epochs=config['training']['n_epochs'],callbacks = cb)
