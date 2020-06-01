# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:17
# @Author wcy
import os
import time

import tensorflow as tf
import numpy as np

from config import USER_MODEL_DATA_PATH
from dataset import Dataset
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    batch = 256
    dataset = Dataset(batch=batch)
    model = Model()
    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        saver.restore(sess, tf.train.latest_checkpoint(USER_MODEL_DATA_PATH))
        while True:
            batch_data = dataset.train_next()
            feed_dict = {
                model.input_query_vector: batch_data["query_vector"],
                model.input_boxes_mask: batch_data["boxes_mask"],
                model.input_feature_vector: batch_data["feature_vector"],
                model.input_boxe_vector: batch_data["boxe_vector"],
                model.input_label_vector: batch_data["label_vector"],
                model.input_labels: batch_data["triplet_labels"],
            }
            _, gs_num = sess.run([model.train_op, model.global_step], feed_dict=feed_dict)
            if gs_num % 10 == 0:
                total_loss, all_loss, hard_loss = sess.run(
                    [model.loss, model.all_triplet_loss, model.hard_triplet_loss], feed_dict=feed_dict)
                print(f"{gs_num}: total_loss={total_loss} all_loss={all_loss} hard_loss={hard_loss}")
            if gs_num % 1000 == 0:
                saver.save(sess, os.path.join(USER_MODEL_DATA_PATH, 'model'),
                           global_step=model.global_step)
