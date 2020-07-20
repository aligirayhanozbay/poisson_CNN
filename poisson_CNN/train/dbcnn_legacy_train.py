import argparse, json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .utils import choose_optimizer, load_model_checkpoint
from ..models import Dirichlet_BC_NN_Legacy_2
from ..losses import loss_wrapper
from ..dataset.generators import numerical_dataset_generator



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

model = Dirichlet_BC_NN_Legacy_2(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(**config['training']['loss_parameters'])
dataset = numerical_dataset_generator(randomize_boundary_smoothness = True, exclude_zero_boundaries = True, nonzero_boundaries = ['left'], rhses = 'zero', return_boundaries = True, return_dx = True, return_rhs = False, **config['dataset'])

inp,tar=dataset.__getitem__(0)
out = model(inp + [tf.shape(tar)[2]])
model.compile(loss=loss,optimizer=optimizer)
cb = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt.checkpoint',save_weights_only=True,save_best_only=True,monitor = 'loss'),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
    tf.keras.callbacks.TerminateOnNaN()
]

load_model_checkpoint(model, args.continue_from_checkpoint, model_config = config['model'], sample_model_input = inp)
model.summary()

model.fit(dataset,epochs=config['training']['n_epochs'],callbacks = cb)
