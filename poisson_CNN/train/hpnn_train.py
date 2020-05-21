import tensorflow as tf
import argparse, json

from ..models import Homogeneous_Poisson_NN
from ..losses import loss_wrapper
from ..dataset.generators import reverse_poisson_dataset_generator

def choose_optimizer(name):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam
    elif name == "sgd":
        return tf.keras.optimizers.SGD

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")

args = parser.parse_args()


config = json.load(open(args.config))
checkpoint_dir = args.checkpoint_dir

for key in config['model'].keys():
    if 'config' in key:
        for layer_config_key in config['model'][key].keys():
            if 'activation' in layer_config_key:
                config['model'][key][layer_config_key] = eval(config['model'][key][layer_config_key])

model = Homogeneous_Poisson_NN(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(**config['training']['loss_parameters'])

dataset = reverse_poisson_dataset_generator(**config['dataset'])

inp,_=dataset.__getitem__()
out = model(inp)

model.compile(loss=loss,optimizer=optimizer)
cb = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt.checkpoint',save_weights_only=True,save_best_only=True,monitor = 'loss'),
    tf.keras.callbacks.ReduceLROnPlateau(patience = 2,monitor='loss')
]

model.fit(dataset,epochs=50,callbacks = cb)
