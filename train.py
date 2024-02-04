# -*- coding=utf-8 -*-
# tensorflow=1.15.0

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
import tqdm

from second_order_fgnn import GCNPolicy as SecondOrderFGNN
from utils import load_data_folder
# TODO: import classic GNN


MODEL_DICT = {
    'SecondOrderFGNN': SecondOrderFGNN,
    'GNN': None, #TODO
}

parser = argparse.ArgumentParser()
# parser.add_argument("--data", type=int, default=100,
#                     help="number of training data")
parser.add_argument("--gpu", type=int, default=0,
                    help="index of the gpu used for training")
parser.add_argument("--emb-size", type=int, default=6,
                    help="embedding size of hidden states in 2-FGNN")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="initial learning rate")
parser.add_argument("--num-epochs", type=int, default="10000",
                    help="num of epochs for training")
parser.add_argument("--data-path", type=str,
                    help="path to the directory that contains training data")
parser.add_argument("--model", type=str, choices=['SecondOrderFGNN', 'GNN'],
                    help="type of model that is trained")
parser.add_argument("--seed", type=int, default=1812,
                    help="random seed for reproducibility")
parser.add_argument("--save-path", type=str, default='./results/default',
                    help="path where checkpoints and logs are saved")


def process(model, dataset, optimizer):
    """Train the network for one epoch.
    """
    (cons_features, edge_indices, edge_features,
     var_features, branch_scores)  = dataset

    num_samples = cons_features.shape[0]
    order = np.random.shuffle(np.arange(num_samples, dtype=int))

    accumulated_loss = 0.0
    with tf.GradientTape() as tape:
        for i in tqdm.tqdm(order):
            inputs = (cons_features[i], edge_indices[i], edge_features[i],
                      var_features[i])
            out = model(inputs, training=True)

            loss = tf.keras.metrics.mean_squared_error(branch_scores[i], out)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            accumulated_loss += loss.numpy()

    return accumulated_loss / num_samples


if __name__ == "__main__":
    args = parser.parse_args()

    ## Set up dataset
    (var_features, cons_features, edge_features, edge_indices,
     branch_scores, num_vars, num_conss, num_edges,
     var_dim, cons_dim, edge_dim) = load_data_folder(args.data_path)

    ## Set up model
    os.makedirs(args.save_path, exist_ok=True)
    model_save_path = os.path.joint(args.save_path, 'model.pkl')

    # Set up TensorFlow Eager mode
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution(config)
    tf.executing_eagerly()
    # Set up GPU device
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed+1)
    with tf.device("GPU:"+str(args.gpu)):
        var_features = tf.constant(var_features, dtype=tf.float32)
        cons_features = tf.constant(cons_features, dtype=tf.float32)
        edge_features = tf.constant(edge_features, dtype=tf.float32)
        edge_indices = tf.constant(edge_indices, dtype=tf.int32)
        branch_scores = tf.constant(branch_scores, dtype=tf.float32)
        train_data = (cons_features, edge_indices, edge_features,
                      var_features, branch_scores)

        # Initialization
        model = MODEL_DICT[args.model](args.emb_size, cons_dim, edge_dim, var_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        loss_best = 1e10

        ### MAIN LOOP ###
        # while epoch <= args.num_epochs:
        for epoch in args.num_epochs:
            train_loss = process(model, train_data, optimizer)

            print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}")
            if train_loss < loss_best:
                model.save_state(model_save_path)
                print("model saved to:", model_save_path)
                loss_best = train_loss

        model.summary()
