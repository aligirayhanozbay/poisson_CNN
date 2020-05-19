import tensorflow as tf
import argparse, json

from ..model import Homogeneous_Poisson_NN
from ..losses import loss_wrapper

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

config = json.load(args['config'])
checkpoint_dir = args['checkpoint_dir']

model = Homogeneous_Poisson_NN(**config['model'])
optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
loss = loss_wrapper(**config['training']['loss_parameters'])
