import argparse, json, os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['NCCL_DEBUG'] = 'INFO'
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from .utils import load_model_checkpoint, choose_optimizer
from ..utils import convert_tf_object_names
from ..models import Homogeneous_Poisson_NN_Legacy, Dirichlet_BC_NN_Legacy_2, Poisson_CNN_Legacy
from ..losses import loss_wrapper
from ..dataset.generators import numerical_dataset_generator, reverse_poisson_dataset_generator

parser = argparse.ArgumentParser(description="Train the Homogeneous Poisson NN")
parser.add_argument("config", type=str, help="Path to the configuration json for training, model and dataset parameters")
parser.add_argument("--checkpoint_dir", type=str, help="Directory to save result checkpoints in", default=".")
parser.add_argument("--continue_from_checkpoint", type=str, help="Continue from this checkpoint file if provided", default=None)
parser.add_argument("--learning_rate", type=str, help="Overrides the learning rate with the provided value, or with the value from the json file if 'from_json' is the provided value", default = None)

args = parser.parse_args()

config = convert_tf_object_names(json.load(open(args.config)))
checkpoint_dir = args.checkpoint_dir

if 'precision' in config['training'].keys():
    tf.keras.backend.set_floatx(config['training']['precision'])

dataset = numerical_dataset_generator(randomize_boundary_smoothness = True, exclude_zero_boundaries = False, nonzero_boundaries = ['left','right','top','bottom'], rhses = 'random', return_boundaries=True, return_dx = True, return_rhs = True, **config['dataset'])

dist_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())#nccl causes errors on POWER9
with dist_strategy.scope():
    hpnn = Homogeneous_Poisson_NN_Legacy(**config['hpnn_model'])
    dbcnn = Dirichlet_BC_NN_Legacy_2(**config['dbcnn_model'])
    model = Poisson_CNN_Legacy(hpnn,dbcnn)
    optimizer = choose_optimizer(config['training']['optimizer'])(**config['training']['optimizer_parameters'])
    loss = loss_wrapper(global_batch_size = config['dataset']['batch_size'], **config['training']['loss_parameters'])

    
    inp,tar=dataset.__getitem__(0)
    out = model([x[:1] for x in inp])
    model.compile(loss=loss,optimizer=optimizer)
    cb = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + '/chkpt.checkpoint',save_weights_only=True,save_best_only=True,monitor = 'loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience = 4,monitor='loss',min_lr=config['training']['min_learning_rate']),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    #load_model_checkpoint(model, args.continue_from_checkpoint, model_config = config['model'], sample_model_input = inp)

    if args.learning_rate is not None:
        model.optimizer.learning_rate = config['training']['optimizer_parameters']['learning_rate'] if args.learning_rate.lower() == 'from_json' else float(args.learning_rate)
    
    model.summary()
    model.run_eagerly = True
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model.fit(dataset,epochs=config['training']['n_epochs'],callbacks = cb)
