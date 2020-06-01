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

    def __init__(self, max_boxes_num=10, lr=0.0005, global_step=0):
        super().__init__()
        self.max_boxes_num = max_boxes_num
        self.lr = lr
        self.global_step = tf.Variable(global_step, trainable=False)
        self.margin = 0.5
        self.input_query_vector = tf.placeholder(tf.float32, (None, 768))  # shape (batch, 768)

        self.input_boxes_mask = tf.placeholder(tf.float32, (None, self.max_boxes_num))
        self.input_feature_vector = tf.placeholder(tf.float32, (None, self.max_boxes_num, 2048))  # shape (batch, boxe_num, 2048)
        self.input_boxe_vector = tf.placeholder(tf.float32, (None, self.max_boxes_num, 3))  # shape (batch, boxe_num, (x, y, area_than)
        self.input_label_vector = tf.placeholder(tf.float32, (None, self.max_boxes_num, 768))  # shape (batch, boxe_num, 768)

        self.input_labels = tf.placeholder(tf.int32, (None))  # shape (batch*2)
        self.build()

    def build(self):
        self.image_feature = self.build_image_feature()
        self.query_feature = self.build_query_feature()
        self.embeddings = tf.concat((self.query_feature, self.image_feature), axis=0)
        self.all_triplet_loss, _ = self.batch_all_triplet_loss(self.input_labels, self.embeddings, self.margin)
        self.hard_triplet_loss = self.batch_hard_triplet_loss(self.input_labels, self.embeddings, self.margin)
        self.loss = (tf.where(tf.greater_equal(self.global_step, 3000), self.hard_triplet_loss, self.all_triplet_loss))
        self.build_gradient_descent()

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
            inputs = tf.expand_dims(self.input_boxes_mask, axis=-1)*self.input_feature_vector
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same")(inputs)
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
            self.box_feature = tf.reshape(layers, (-1, 768))

        with tf.variable_scope(None, "label_feature"):
            filters_kernel_size = [[1024, 3], [512, 3], [1024, 2], [512, 2]]
            layers = []
            inputs = tf.expand_dims(self.input_boxes_mask, axis=-1) * self.input_label_vector
            for f, k in filters_kernel_size:
                x = KL.Conv1D(f, k, padding="same")(inputs)
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
            self.label_feature = tf.reshape(layers, (-1, 768))

        with tf.variable_scope(None, "feature_merge"):
            layers = tf.concat((self.label_feature, self.box_feature), axis=-1)
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

    def build_gradient_descent(self):
        # 学习率连续衰减
        starter_learning_rate = self.lr
        end_learning_rate = 0.000005
        decay_rate = 0.01
        start_decay_step = 1500
        decay_steps = 10000  #
        learning_rate = (
            tf.where(
                tf.greater_equal(self.global_step, start_decay_step),  # if global_step >= start_decay_step
                # 具体选择那个衰减函数，请查看decay.py绘制的曲线
                tf.train.polynomial_decay(starter_learning_rate, self.global_step - start_decay_step, decay_steps,
                                          end_learning_rate, power=1.0),
                # tf.train.exponential_decay(starter_learning_rate, global_step - start_decay_step, decay_steps=decay_steps, decay_rate=decay_rate),
                # tf.train.inverse_time_decay(starter_learning_rate, global_step - start_decay_step, decay_steps=decay_steps, decay_rate=decay_rate),
                # tf.train.natural_exp_decay(starter_learning_rate, global_step - start_decay_step, decay_steps=decay_steps, decay_rate=decay_rate),
                starter_learning_rate
            )
        )

        with tf.variable_scope(None, "optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            self.train_op = optimizer.minimize(self.loss, self.global_step, colocate_gradients_with_ops=True)
        self.merged_summary_op = tf.summary.merge_all()



if __name__ == '__main__':
    batch = 32
    model = Model()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while True:
            box_num = 10
            feed_dict = {
                model.input_query_vector: np.random.random((batch, 768)),
                model.input_boxes_mask: np.random.randint(0, 2, (batch, box_num)),
                model.input_feature_vector: np.random.random((batch, box_num, 2048)),
                model.input_boxe_vector: np.random.random((batch, box_num, 3)),
                model.input_label_vector: np.random.random((batch, box_num, 768)),
                model.input_labels: np.random.randint(0, 10, (batch*2)),
            }
            s = time.time()
            a = sess.run([model.all_triplet_loss, model.hard_triplet_loss], feed_dict=feed_dict)
            e = time.time()
            print(f"{e-s}", a)
