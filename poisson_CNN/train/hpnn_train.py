import tensorflow as tf
import argparse, json

from .utils import load_model_checkpoint, choose_optimizer
from ..utils import convert_tf_object_names
from ..models import Homogeneous_Poisson_NN_Metalearning, Homogeneous_Poisson_NN, Homogeneous_Poisson_NN_Autoencoder_2D
from ..losses import loss_wrapper
from ..dataset.generators import reverse_poisson_dataset_generator

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided", default=None)

args = parser.parse_args()


config = convert_tf_object_names(json.load(open(args.config)))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

model_type = config['model'].pop('model_type')
if model_type == 'cnn_metalearning':
    model = Homogeneous_Poisson_NN_Metalearning(**config['model'])
elif model_type == 'cnn':
    model = Homogeneous_Poisson_NN(**config['model'])
elif model_type == 'autoencoder':
    model = Homogeneous_Poisson_NN_Autoencoder_2D(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(**config['training']['loss_parameters'])


dataset = reverse_poisson_dataset_generator(**config['dataset'])

inp,tar=dataset.__getitem__()
out = model(inp)

model.compile(loss=loss,optimizer=optimizer)
cb = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt.checkpoint',save_weights_only=True,save_best_only=True,monitor = 'loss'),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
    tf.keras.callbacks.TerminateOnNaN()
]

load_model_checkpoint(model, args.continue_from_checkpoint, model_config = config['model'], sample_model_input = inp)

model.summary()
#model.run_eagerly = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
model.fit(dataset,epochs=config['training']['n_epochs'],callbacks = cb)
