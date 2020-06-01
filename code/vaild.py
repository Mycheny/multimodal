# -*- coding: utf-8 -*- 
# @Time 2020/6/1 15:41
# @Author wcy
import base64
import json
import os

import pandas as pd
import tensorflow as tf
import numpy as np

from bert.bert2vec import BertEncode
from config import USER_MODEL_DATA_PATH, VALID_PATH, LABELS_NAMES_PATH, VALID_ANSWER_PATH
from dataset import Dataset
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

labels = pd.read_csv(LABELS_NAMES_PATH, sep="\t").values[:, 1]


def _pairwise_distances(v1, v2, squared=False):
    embeddings = np.concatenate((v1, v2), axis=0)
    dot_product = np.matmul(embeddings, np.transpose(embeddings))
    square_norm = np.diag(dot_product)
    distances = np.expand_dims(square_norm, 0) - 2.0 * dot_product + np.expand_dims(square_norm, 1)
    distances = np.maximum(distances, 0.0)  # 类似 np.sum(np.square(embeddings[0] - embeddings[1]))
    if not squared:
        mask = (np.equal(distances, 0.0)).astype(np.float32)
        distances = distances + mask * 1e-16
        distances = np.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.


def get_ndcg(r, ref, k):
    dcg_max = dcg_at_k(ref, k)
    if not dcg_max:
        return 0.
    dcg = dcg_at_k(r, k)
    return dcg / dcg_max


def evaluate(predictions, reference):
    k = 5
    predictions = {k: [str(v) for v in vs] for k, vs in predictions.items()}
    # compute score for each query
    score_sum = 0.
    for qid in reference.keys():
        ground_truth_ids = set([str(pid) for pid in reference[qid]])
        ref_vec = [1.0] * len(ground_truth_ids)
        pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions[qid]]
        score_sum += get_ndcg(pred_vec, ref_vec, k)
    # the higher score, the better
    score = score_sum / len(reference)
    return score


if __name__ == '__main__':
    valid_answer = json.load(open(VALID_ANSWER_PATH, "r"))
    batch = 256
    text2vec = BertEncode(graph_path=None)
    dataset = Dataset(batch=batch)
    model = Model()
    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(USER_MODEL_DATA_PATH))

        valid_datas = pd.read_csv(VALID_PATH, sep="\t")
        for name, group in valid_datas.groupby("query_id"):
            distances = {}
            for index, tup in enumerate(group.itertuples()):
                product_id = tup.product_id
                image_h = tup.image_h
                image_w = tup.image_w
                num_boxes = tup.num_boxes
                boxes = np.frombuffer(base64.b64decode(tup.boxes), dtype=np.float32).reshape(num_boxes, 4)
                features = np.frombuffer(base64.b64decode(tup.features), dtype=np.float32).reshape(num_boxes, 2048)
                class_labels = np.frombuffer(base64.b64decode(tup.class_labels), dtype=np.int64).reshape(num_boxes)
                class_labels_names = [labels[index] for index in class_labels]
                query = tup.query
                query_id = tup.query_id

                query_vector = text2vec.encode([query])

                class_labels_vector = text2vec.encode(class_labels_names)
                boxe_vector = dataset.build_boxe_vector(np.array([image_h]), np.array([image_w]), np.array([boxes]))
                boxes_mask = dataset.build_mask(np.array([num_boxes]))
                boxe_vector = dataset.expand_zeros(boxe_vector)
                feature_vector = dataset.expand_zeros(np.expand_dims(features, axis=0))
                label_vector = dataset.expand_zeros(np.expand_dims(class_labels_vector, axis=0))

                feed_dict = {
                    model.input_query_vector: query_vector,
                    model.input_boxes_mask: boxes_mask,
                    model.input_feature_vector: feature_vector,
                    model.input_boxe_vector: boxe_vector,
                    model.input_label_vector: label_vector
                }
                query_feature, image_feature = sess.run([model.query_feature, model.image_feature], feed_dict=feed_dict)
                dist = _pairwise_distances(query_feature, image_feature)[0][1]
                distances[product_id] = dist
            label_top5 = valid_answer[str(name)]
            distances = sorted(distances.items(), key=lambda d: d[1], reverse=False)
            score = evaluate({"a": [i[0] for i in distances[:5]]}, {"a": label_top5})
            print()
