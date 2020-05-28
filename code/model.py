# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:17
# @Author wcy
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
        self.input_query_vector = tf.placeholder(tf.float32, (None, 768))  # shape (batch, 768)

        self.input_feature_vector = tf.placeholder(tf.float32, (None, None, 2048))  # shape (batch, boxe_num, 2048)
        self.input_boxe_vector = tf.placeholder(tf.float32, (None, None, 3))  # shape (batch, boxe_num, (x, y, area_than)
        self.input_label_vector = tf.placeholder(tf.float32, (None, None, 768))  # shape (batch, boxe_num, 768)
        self.build()

    def build(self):
        x = KL.Conv2D(128, (3, 3), padding="valid")(tf.zeros((32, 28, 28, 3), dtype=tf.float32))
        dim = 128
        batch = 5
        labels = np.random.randint(0, 5, (batch,))
        embeddings = (np.round(np.random.random((batch, dim)), 1)).astype(np.float32)
        margin = 0.5
        all_triplet_loss = self.batch_all_triplet_loss(labels, embeddings, margin)
        hard_triplet_loss = self.batch_hard_triplet_loss(labels, embeddings, margin)


if __name__ == '__main__':
    Model()