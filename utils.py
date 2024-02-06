import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from pathlib import Path

def get_dense_edge_indices(dim1, dim2):
    """Return a list of edge indices of a dense graph.

    Desc.

    Args:
        - dim1: int
        - dim2: int

    Returns:
        A tensorflow tensor of size (2, dim1 * dim2) with values:
          [[0,0,...,0   ,    1,1,...,1   ,    ......,    dim1,dim1,...,dim1]
           [0,1,...,dim2,    0,1,...,dim2,    ......,    0   ,1   ,...,dim2]]
    """
    r1 = tf.expand_dims(tf.range(dim1),1)
    r2 = tf.expand_dims(tf.range(dim2),1)

    t1 = tf.reshape(tf.tile(r1, [1,dim2]), [-1])
    t2 = tf.reshape(tf.tile(r2, [dim1,1]), [-1])

    return tf.stack([t1,t2], axis=0)


def load_data_folder(dir_path):
    """Load a dataset.

    Read information of MILP problems stored in `dir_path` as sub-directories.
    Note that this function can only process MILP problems with of the same
    numbers of variables, constraints and nonzeros in coefficient matrices.

    The structure of `dir_path` should be:
    dir_path
    |
    |--- problem_0
    |    |-- ConFeatures.csv
    |    |-- EdgeFeatures.csv
    |    |-- EdgeIndices.csv
    |    |-- model.mps
    |    |-- SBScores.csv
    |    |-- VarFeatures.csv
    |
    |--- problem_1
    |    |-- ...
    |
    |--- ...

    Returns:
        - var_features: numpy.ndarray of shape (num_data, num_vars, var_dim)
        - cons_features: numpy.ndarray of shape (num_data, num_conss, cons_dim)
        - edge_features: numpy.ndarray of shape (num_data, num_edges, edge_dim)
        - edge_indices: numpy.ndarray of shape (num_data, 2, num_edges)
        - branch_scores: numpy.ndarray of shape (num_data, num_vars)
        - num_vars: int, the number of variables in each MILP problem
        - num_conss: int, the number of constraints in each MILP problem
        - num_edges: int, the number of edges in each MILP problem
        - var_dim: int, the dimension of the variable features
        - cons_dim: int, the dimension of the constraint features
        - edge_dim: int, the dimension of the edge features
    """
    dir_path = Path(dir_path)


    (var_features, cons_features, edge_features, edge_indices,
     branch_scores) = [],[],[],[],[]
    for prob in dir_path.glob("*"):
        var_features.append(np.loadtxt(prob/'VarFeatures.csv', delimiter=',', ndmin=2))
        cons_features.append(np.loadtxt(prob/'ConFeatures.csv', delimiter=',', ndmin=2))
        edge_features.append(np.loadtxt(prob/'EdgeFeatures.csv', delimiter=',', ndmin=2))
        edge_indices.append(np.loadtxt(prob/'EdgeIndices.csv', delimiter=',', ndmin=2))
        branch_scores.append(np.loadtxt(prob/'SBScores.csv', delimiter=','))

    var_features = np.stack(var_features).astype(np.float32)
    cons_features = np.stack(cons_features).astype(np.float32)
    edge_features = np.stack(edge_features).astype(np.float32)
    edge_indices = np.stack(edge_indices).astype(np.int32).transpose(0,2,1)
    branch_scores = np.stack(branch_scores).astype(np.float32)
    num_data, num_vars, var_dim = var_features.shape
    _, num_conss, cons_dim = cons_features.shape
    _, num_edges, edge_dim = edge_features.shape

    return (var_features, cons_features, edge_features, edge_indices,
            branch_scores, num_vars, num_conss, num_edges,
            var_dim, cons_dim, edge_dim)


class SecondOrderFGNNConvolution(K.Model):
    """A Variant of Second-order Folklore GNN convolution layer for testing.

    This is a variant of Second-order Folklore GNN convolution layer, which does
    not contain feature transformation layers but only feature concatenations.
    This class is solely used for testing its correctness and not used in the
    2-FGNN implementation.
    """
    def __init__(self, emb_size, activation, initializer):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer

        self.s_update_layer = K.Sequential([])
        self.t_update_layer = K.Sequential([])

        self.s_output_layer = K.Sequential([])
        self.t_output_layer = K.Sequential([])

    def build(self, input_shapes):
        self.built = True

    def call(self, inputs):
        s_prev, t_prev = inputs
        num_conss, num_vars, _ = s_prev.shape

        s1_tiled = tf.tile(tf.expand_dims(s_prev, axis=1), [1,num_vars,1,1])
        s2_tiled = tf.tile(tf.expand_dims(s_prev, axis=2), [1,1,num_vars,1])
        t_tiled = tf.tile(tf.expand_dims(t_prev, axis=0), [num_conss,1,1,1])

        st_joint = tf.concat([t_tiled, s2_tiled], axis=-1)
        st_transformed = self.s_update_layer(st_joint)
        s_update = tf.reduce_sum(st_transformed, axis=1)
        s_features = self.s_output_layer(tf.concat([s_prev, s_update], axis=-1))

        ss_joint = tf.concat([s1_tiled, s2_tiled], axis=-1)
        ss_transformed = self.t_update_layer(ss_joint)
        t_update = tf.reduce_sum(ss_transformed, axis=0)
        t_features = self.t_output_layer(tf.concat([t_prev, t_update], axis=-1))

        return s_features, t_features


def second_order_fgnn_conv_numpy(s_input, t_input):
    """Naive implementation of the convolution operation on 2-FGNN with Numpy.

    This is an naive implementation of the convolution operation of 2-FGNN
    usnig loops to make sure the outputs are correct. We do not apply MLP
    transformations on the features but only concatenations. This function is
    used to verify the correctness of the Tensorflow implementation.
    """
    num_conss, num_vars, s_emb = s_input.shape
    _, _, t_emb = t_input.shape
    s_output = np.zeros(shape=(num_conss, num_vars, s_emb*2 + t_emb))
    t_output = np.zeros(shape=(num_vars, num_vars,  s_emb*2 + t_emb))

    for i in range(num_conss):
        for j in range(num_vars):
            s = s_input[i,j]
            ts = np.zeros(shape=[t_emb + s_emb])
            for j1 in range(num_vars):
                ts += np.concatenate([t_input[j1,j], s_input[i,j1]])
            s_output[i,j] = np.concatenate([s, ts])

    for j1 in range(num_vars):
        for j2 in range(num_vars):
            t = t_input[j1,j2]
            ss = np.zeros(shape=[s_emb*2])
            for i in range(num_conss):
                ss += np.concatenate([s_input[i,j2], s_input[i,j1]])
            t_output[j1,j2] = np.concatenate([t,ss])

    return s_output, t_output


if __name__ == "__main__":
    sess = tf.InteractiveSession()

    s_input = np.random.normal(size=(3,4,8))
    t_input = np.random.normal(size=(4,4,8))

    s_output_numpy, t_output_numpy = second_order_fgnn_conv_numpy(s_input, t_input)

    conv = SecondOrderFGNNConvolution(8, K.activations.relu, K.initializers.orthogonal)
    conv.build(((None, None, 8), (None, None, 8)))
    s_output_tf, t_output_tf = conv((tf.convert_to_tensor(s_input),
                                     tf.convert_to_tensor(t_input)))

    print(np.allclose(s_output_tf.eval(), s_output_numpy))
    print(np.allclose(t_output_tf.eval(), t_output_numpy))
