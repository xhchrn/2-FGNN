import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as K

from utils import get_dense_edge_indices


class SecondOrderEmbeddingLayer(K.Model):
    """Compute the ebmeddings for 2FGNN networks.
    """

    def __init__(self, emb_size, activation, initializer):
        super().__init__()
        self.emb_size= emb_size
        self.activation = activation
        self.initializer = initializer

        self.embedding_layer = K.layers.Dense(units=self.emb_size,
                                              activation=self.activation,
                                              kernel_initializer=self.initializer)

    def build(self, input_shapes):
        first_shape, second_shape, ef_shape, _ = input_shapes
        joint_shape = [None, first_shape[1] + second_shape[1] + ef_shape[1]]
        self.embedding_layer.build(joint_shape)
        self.built = True

    def call(self, first_feats, second_feats, edge_feats, edge_indices):

        dense_indices = get_dense_edge_indices(first_feats.shape[0],
                                               second_feats.shape[0])
        first_gathered = tf.gather(first_feats, axis=0, indices=dense_indices[0])
        second_gathered = tf.gather(second_feats, axis=0, indices=dense_indices[1])

        scatter_indices = edge_indices[0] * first_feats.shape[0] + edge_indices[1]
        edge_gathered = tf.scatter_nd(
            updates=edge_feats,
            indices=tf.expand_dims(scatter_indices, axis=1),
            shape=[dense_indices.shape[1], 1]
        )

        joint_features = tf.concat(
            [first_gathered, second_gathered, edge_gathered], axis=1)

        output = self.embedding_layer(joint_features)
        output = tf.reshape(output, (first_feats.shape[0], second_feats.shape[0], -1))
        return output


class SecondOrderFGNNConvolution(K.Model):
    """Second-order Folklore GNN convolution layer.
    """
    def __init__(self, emb_size, activation, initializer):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer

        self.s_update_layer = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])
        self.t_update_layer = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])

        self.s_output_layer = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])
        self.t_output_layer = K.Sequential([
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
            K.layers.Activation(self.activation),
            K.layers.Dense(units=self.emb_size, kernel_initializer=self.initializer),
        ])

    def build(self, input_shapes):
        s_shape, t_shape = input_shapes
        self.s_update_layer.build([None, None, None, t_shape[-1] + s_shape[-1]])
        self.t_update_layer.build([None, None, None, s_shape[-1] * 2])
        self.s_output_layer.build([None, None, s_shape[-1] + self.emb_size])
        self.t_output_layer.build([None, None, t_shape[-1] + self.emb_size])
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


class GCNPolicy(K.Model):
    """Desc
    """
    def __init__(self, embSize, nConsF, nEdgeF, nVarF):
        super().__init__()

        self.emb_size = embSize
        self.cons_nfeats = nConsF
        self.edge_nfeats = nEdgeF
        self.var_nfeats = nVarF

        self.activation = K.activations.relu
        self.initializer = K.initializers.Orthogonal()

        # Embeddings
        self.s_embedding = SecondOrderEmbeddingLayer(self.emb_size,
                                                     self.activation,
                                                     self.initializer)
        self.t_embedding = SecondOrderEmbeddingLayer(self.emb_size,
                                                     self.activation,
                                                     self.initializer)

        # Graph convolutions
        self.conv_st_1 = SecondOrderFGNNConvolution(self.emb_size,
                                                    self.activation,
                                                    self.initializer)
        self.conv_st_2 = SecondOrderFGNNConvolution(self.emb_size,
                                                    self.activation,
                                                    self.initializer)

        # Output
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.emb_size,
                           activation=self.activation,
                           kernel_initializer=self.initializer),
            K.layers.Dense(units=1,
                           activation=None,
                           kernel_initializer=self.initializer,
                           use_bias=False),
        ])

        # build model right-away
        self.build([(None, self.cons_nfeats),
                    (2, None),
                    (None, self.edge_nfeats),
                    (None, self.var_nfeats),])

        # save input signature for compilation
        self.input_signature = [
            (
                tf.TensorSpec(shape=[None, self.cons_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[2, None], dtype=tf.int32),
                tf.TensorSpec(shape=[None, self.edge_nfeats], dtype=tf.float32),
                tf.TensorSpec(shape=[None, self.var_nfeats], dtype=tf.float32),
            ),
            tf.TensorSpec(shape=[], dtype=tf.bool),
        ]

        # save / restore fix
        self.variables_topological_order = [v.name for v in self.variables]

    def build(self, input_shapes):
        cons_shape, ei_shape, ef_shape, var_shape = input_shapes
        s_shape = (None, None, self.emb_size)
        t_shape = (None, None, self.emb_size)
        emb_shape = (None, self.emb_size)

        if not self.built:
            self.s_embedding.build((cons_shape, var_shape, ef_shape, ei_shape))
            self.t_embedding.build((var_shape, var_shape, ef_shape, ei_shape))
            self.conv_st_1.build((s_shape, t_shape))
            self.conv_st_2.build((s_shape, t_shape))
            self.output_module.build(emb_shape)
            self.built = True

    def call(self, inputs, training):
        """
        Args:
            - constraint_features: [num_conss, dim_conss]
            - edge_indices: [2, num_nnz]
            - edge_features: [num_nnz, dim_edges]
            - variable features: [num_vars, dim_vars]
        """
        (constraint_features, edge_indices, edge_features,
         variable_features) = inputs

        # Conss-Vars Embeddings
        s_features = self.s_embedding(constraint_features,
                                      variable_features,
                                      edge_features,
                                      edge_indices)

        # Vars-Vars Embeddings
        num_vars = int(variable_features.shape[0])
        d_indices = tf.stack([tf.range(num_vars), tf.range(num_vars)], axis=0)
        d_feats = tf.ones(shape=[num_vars, 1], dtype=edge_features.dtype)
        t_features = self.t_embedding(variable_features,
                                      variable_features,
                                      d_feats,
                                      d_indices)

        # Graph convolutions - layer 1
        s_features, t_features = self.conv_st_1(s_features, t_features)
        s_features = self.activation(s_features)
        t_features = self.activation(t_features)
        # Graph convolutions - layer 2
        s_features, t_features = self.conv_st_2(s_features, t_features)
        s_features = self.activation(s_features)
        t_features = self.activation(t_features)

        # Summation of features over variables
        joint_features = tf.concat([tf.reduce_sum(s_features, axis=0),
                                    tf.reduce_sum(t_features, axis=0)],
                                   axis=1)
        output = self.output_module(joint_features)
        return output

    def save_state(self, path):
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    def restore_state(self, path):
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    net = GCNPolicy(8, 2, 1 ,4)
    conss_feats = tf.random_normal(shape=(3,2))
    edge_indices = tf.cast(tf.random_uniform(shape=(2,7))*5, tf.int32)
    edge_feats = tf.random_normal(shape=(7,1))
    var_feats = tf.random_normal(shape=(5,4))
    out = net((conss_feats, edge_indices, edge_feats, var_feats), False)
    print(out)
