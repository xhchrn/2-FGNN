# -*- coding=utf-8 -*-
# tensorflow=1.15.0

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm

from gnn import GNN
from pathlib import Path
from second_order_fgnn import SecondOrderFGNN
from utils import load_data_folder


MODEL_DICT = {
    'GNN': GNN,
    'SecondOrderFGNN': SecondOrderFGNN,
}

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="index of the gpu used for training")
parser.add_argument("--emb-size", type=int, default=6,
                    help="embedding size of hidden states in 2-FGNN")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="initial learning rate")
parser.add_argument("--lr-decay", type=str, default=None,
                    help="learning rate scheduling")
parser.add_argument("--num-epochs", type=int, default="10000",
                    help="num of epochs for training")
parser.add_argument("--data-path", type=str,
                    help="path to the directory that contains training data")
parser.add_argument("--model", type=str, choices=['SecondOrderFGNN', 'GNN'],
                    help="type of model that is trained")
parser.add_argument("--optimizer", type=str, default='Adam', choices=['Adam', 'SGD'],
                    help="type of optimizer to use")
parser.add_argument("--seed", type=int, default=1812,
                    help="random seed for reproducibility")
parser.add_argument("--save-path", type=str, default='./results/default',
                    help="path where checkpoints and logs are saved")
parser.add_argument("--best-wait", type=int, default=-1,
                    help="number of epochs of waiting for better training loss")


def setup_logger(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logdir = Path(logdir)
    logging.basicConfig(
        format="[%(asctime)s] [%(name)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(logdir/'log.txt', mode='w'),
                  logging.StreamHandler(os.sys.stdout)]
    )
    return logging.getLogger('main')


def process(model, dataset, optimizer):
    """Train the network for one epoch.
    """
    (cons_features, edge_indices, edge_features,
     var_features, branch_scores)  = dataset

    num_samples = len(cons_features)
    order = np.arange(num_samples, dtype=int)
    np.random.shuffle(order)

    train_vars = model.variables
    accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
    accumulated_loss = 0.0
    for i in tqdm.tqdm(order):
        with tf.GradientTape() as tape:
            inputs = (cons_features[i], edge_indices[i], edge_features[i],
                      var_features[i])
            out = model(inputs, training=True)

            loss = tf.keras.metrics.mean_squared_error(branch_scores[i], out)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(target=loss, sources=train_vars)

            accum_gradient = [(accum_grad + grad)
                              for accum_grad,grad in zip(accum_gradient, grads)]
            accumulated_loss += loss.numpy()

    accum_gradient = [this_grad / num_samples for this_grad in accum_gradient]
    # import ipdb; ipdb.set_trace(context=10)
    optimizer.apply_gradients(zip(accum_gradient, train_vars))

    print(out.numpy())
    print(branch_scores[i].numpy())

    return accumulated_loss / num_samples


if __name__ == "__main__":
    args = parser.parse_args()

    ## Set up model
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(args.save_path)
    model_save_path = os.path.join(args.save_path, 'model.pkl')

    # Set up TensorFlow Eager mode
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution(config)
    tf.executing_eagerly()
    # Set up GPU device
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed+1)
    with tf.device("GPU:"+str(args.gpu)):
        ## Set up dataset
        (var_features, cons_features, edge_features, edge_indices, branch_scores,
         var_dim, cons_dim, edge_dim) = load_data_folder(args.data_path)
        train_data = (cons_features, edge_indices, edge_features,
                      var_features, branch_scores)

        # Initialization
        model = MODEL_DICT[args.model](args.emb_size, cons_dim, edge_dim, var_dim)
        #if args.lr_decay is None:
        lr = args.lr
        # elif args.lr_decay == 'exponential':
        #     lr = K.optimizers.schedules.ExponentialDecay(
        #         args.lr, decay_steps=500, decay_rate=0.9, staircase=True)
        # else:
        #     raise NotImplementedError
        if args.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif args.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        decay_1 = decay_2 = decay_3 = False
        loss_best = 1e10
        best_epoch = -1

        ### MAIN LOOP ###
        for epoch in range(args.num_epochs):
            train_loss = process(model, train_data, optimizer)

            logger.info(f"Epoch: {epoch}, Train loss: {train_loss}")
            if train_loss < loss_best:
                # model.save_state(model_save_path)
                # logger.info(f"Model checkpoint saved to: {model_save_path}")
                loss_best = train_loss
                best_epoch = epoch
            elif args.best_wait > 0 and epoch - best_epoch >= args.best_wait:
                lr *= 0.5
                logger.info(f"Loss has not improved for {args.best_wait} epochs. Learning rate decayed to {lr}")
                optimizer.lr.assign(new_lr)

            if not decay_1 and train_loss < 1e-6:
                new_lr = lr * 0.1
                logger.info(f"Loss reached 1e-6. Learning rate decayed to {new_lr}")
                optimizer.lr.assign(new_lr)
                decay_1 = True
            if not decay_2 and train_loss < 1e-12:
                new_lr = lr * 0.05
                logger.info(f"Loss reached 1e-12. Learning rate decayed to {new_lr}")
                optimizer.lr.assign(new_lr)
                decay_2 = True
            if not decay_3 and train_loss < 1e-14:
                new_lr = lr * 0.001
                logger.info(f"Loss reached 1e-14. Learning rate decayed to {new_lr}")
                optimizer.lr.assign(new_lr)
                decay_3 = True

        model.summary()
