# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:17
# @Author wcy
import time

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from triplet import Triplet


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class Model(Triplet):

    def __init__(self):
        super().__init__()
        box_num = None
        self.margin = 0.5
        self.input_query_vector = tf.placeholder(tf.float32, (None, 768))  # shape (batch, 768)

        self.input_feature_vector = tf.placeholder(tf.float32, (None, box_num, 2048))  # shape (batch, boxe_num, 2048)
        self.input_boxe_vector = tf.placeholder(tf.float32,
                                                (None, box_num, 3))  # shape (batch, boxe_num, (x, y, area_than)
        self.input_label_vector = tf.placeholder(tf.float32, (None, box_num, 768))  # shape (batch, boxe_num, 768)

        self.input_labels = tf.placeholder(tf.int32, (None))  # shape (batch*2)
        self.build()

    def build(self):
        self.image_feature = self.build_image_feature()
        self.query_feature = self.build_query_feature()
        self.embeddings = tf.concat((self.query_feature, self.image_feature), axis=0)
        self.all_triplet_loss = self.batch_all_triplet_loss(self.input_labels, self.embeddings, self.margin)
        self.hard_triplet_loss = self.batch_hard_triplet_loss(self.input_labels, self.embeddings, self.margin)

    def build_image_feature(self):
        with tf.variable_scope(None, "box_label_weight"):
            filters_kernel_size = [[128, 3], [256, 3], [128, 2], [256, 2]]
            layers = []
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same")(self.input_boxe_vector)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                layers.append(x)
            layers = tf.concat(layers, axis=-1)
            layers = KL.Conv1D(256, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(256, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(128, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1, 3, padding="same")(layers)
            box_label_weight = tf.nn.tanh(layers)

        with tf.variable_scope(None, "box_feature"):
            filters_kernel_size = [[1024, 3], [2048, 3], [1024, 2], [2048, 2]]
            layers = []
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same")(self.input_feature_vector)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                layers.append(x)
            layers = tf.concat(layers, axis=-1)
            layers = KL.Conv1D(2048, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(2048, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1024, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(768, 3, padding="same")(layers)
            layers = box_label_weight * layers
            layers = [tf.reduce_mean(layers, axis=1, keepdims=True), tf.reduce_sum(layers, axis=1, keepdims=True),
                      tf.reduce_max(layers, axis=1, keepdims=True), tf.reduce_max(layers, axis=1, keepdims=True)]
            layers = tf.concat(layers, axis=1)
            layers = tf.transpose(layers, perm=[0, 2, 1])
            filters_kernel_size = [[2, 5], [2, 3], [1, 5], [1, 3]]
            layers2 = []
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(layers)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                layers2.append(x)
            layers = tf.concat(layers2, axis=-1)
            layers = KL.Conv1D(3, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1, 3, padding="same")(layers)
            box_feature = tf.reshape(layers, (-1, 768))

        with tf.variable_scope(None, "label_feature"):
            filters_kernel_size = [[1024, 3], [512, 3], [1024, 2], [512, 2]]
            layers = []
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same")(self.input_label_vector)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                layers.append(x)
            layers = tf.concat(layers, axis=-1)
            layers = KL.Conv1D(2048, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(2048, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1024, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(768, 3, padding="same")(layers)
            layers = box_label_weight * layers
            layers = [tf.reduce_mean(layers, axis=1, keepdims=True), tf.reduce_sum(layers, axis=1, keepdims=True),
                      tf.reduce_max(layers, axis=1, keepdims=True), tf.reduce_max(layers, axis=1, keepdims=True)]
            layers = tf.concat(layers, axis=1)
            layers = tf.transpose(layers, perm=[0, 2, 1])
            filters_kernel_size = [[2, 5], [2, 3], [1, 5], [1, 3]]
            layers2 = []
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(layers)
                x = KL.Conv1D(f, k, padding="same", activation=K.relu)(x)
                layers2.append(x)
            layers = tf.concat(layers2, axis=-1)
            layers = KL.Conv1D(3, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1, 3, padding="same", activation=K.relu)(layers)
            layers = KL.Conv1D(1, 3, padding="same")(layers)
            label_feature = tf.reshape(layers, (-1, 768))

        with tf.variable_scope(None, "feature_merge"):
            layers = tf.concat((label_feature, box_feature), axis=-1)
            layers = KL.Dense(1024)(layers)
            feature = KL.Dense(768)(layers)
        return feature

    def build_query_feature(self):
        with tf.variable_scope(None, "query_feature"):
            layers = KL.Dense(2048)(self.input_query_vector)
            layers = KL.Dense(1024)(layers)
            layers = KL.Dense(1024)(layers)
            query_feature = KL.Dense(768)(layers)
        return query_feature


if __name__ == '__main__':
    batch = 32
    model = Model()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while True:
            box_num = np.random.randint(0, 10)
            feed_dict = {
                model.input_query_vector: np.random.random((batch, 768)),
                model.input_feature_vector: np.random.random((batch, box_num, 2048)),
                model.input_boxe_vector: np.random.random((batch, box_num, 3)),
                model.input_label_vector: np.random.random((batch, box_num, 768)),
                model.input_labels: np.random.randint(0, 10, (batch*2)),
            }
            s = time.time()
            a = sess.run([model.all_triplet_loss, model.hard_triplet_loss], feed_dict=feed_dict)
            e = time.time()
            print(f"{e-s}", a)
