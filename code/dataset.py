# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:16
# @Author wcy
import base64
import os
import pickle

import tensorflow as tf
import numpy as np

from bert.bert2vec import BertEncode
from config import TRAIN_PATH, VALID_PATH, VALID_ANSWER_PATH, TEST_A_PATH, USER_TMP_DATA_PATH, LABELS_NAMES_PATH
import pandas as pd
from tqdm import tqdm


class Dataset(object):

    def __init__(self, batch=10000):
        self.text2vec = BertEncode(graph_path=None)
        self.labels_dict = pd.read_csv(LABELS_NAMES_PATH, sep="\t").values[:, 1]
        self.labels_vector_dict = self.text2vec.encode(self.labels_dict)
        self.train_datas = pd.read_csv(TRAIN_PATH, sep="\t", chunksize=batch)

    def deal_data(self, data):
        product_id = data.product_id.values
        image_h = data.image_h.values
        image_w = data.image_w.values
        num_boxes = data.num_boxes.values
        boxes = [np.frombuffer(base64.b64decode(boxe), dtype=np.float32).reshape(num_boxe, 4) for num_boxe, boxe in
                 zip(num_boxes, data.boxes)]
        features = [np.frombuffer(base64.b64decode(feature), dtype=np.float32).reshape(num_boxe, 2048) for
                    num_boxe, feature in zip(num_boxes, data.features)]
        class_labels = [np.frombuffer(base64.b64decode(class_label), dtype=np.int64).reshape(num_boxe) for
                        num_boxe, class_label in zip(num_boxes, data.class_labels)]
        class_labels_names = [self.labels_dict[index] for index in class_labels]
        class_labels_vector = [self.labels_vector_dict[index] for index in class_labels]
        query = data["query"].values
        query_vector = [self.text2vec.encode(q) for q in tqdm(query)]
        query_id = data.query_id.values
        return {"product_id": product_id,
                "image_h":image_h,
               "image_w":image_w,
               "num_boxes":num_boxes,
               "boxes":boxes,
               "features":features,
               "class_labels":class_labels,
               "class_labels_names":class_labels_names,
               "class_labels_vector":class_labels_vector,
               "query":query,
               "query_vector":query_vector,
               "query_id":query_id}

    def train_next(self):
        data = next(self.train_datas)
        data = self.deal_data(data)
        query_vector = data["query_vector"]
        feature_vector = data["features"]
        boxe_vector = self.build_boxe_vector(data["image_h"], data["image_w"], data["boxes"])
        label_vector = data["class_labels_vector"]
        triplet_labels = self.build_triplet_labels(data["query_id"], data["query"])
        return

    def build_boxe_vector(self, image_h, image_w, boxes):
        areas = image_h*image_w
        areas_ratio = [[(x2 - x1) * (y2 - y1) / area for x1, y1, x2, y2 in boxe] for boxe, area in zip(boxes, areas)]
        print()

    def build_triplet_labels(self, query_id, query):
        pass


def analyze():
    batch = 10000
    dataset = Dataset(batch=batch)
    pkl_file_path = os.path.join(USER_TMP_DATA_PATH, "train.pkl")
    if not os.path.exists(pkl_file_path):
        # columns = ["product_id", "image_h", "image_w", "num_boxes", "query", "query_id"]
        columns = ["product_id", "query", "query_id"]
        datas = pd.DataFrame(columns=columns)
        for data in tqdm(dataset.train_datas):
            data = data[columns]
            # res = [i for i in list(data.groupby("query_id"))]
            datas = pd.concat((datas, data))
        del data
        with open(pkl_file_path, "wb") as f:
            pickle.dump(datas, f)
    else:
        with open(pkl_file_path, "rb") as f:
            datas = pickle.load(f)

    tmp = {name: value.shape[0] for name, value in tqdm(datas.groupby("query_id"))}


if __name__ == '__main__':
    batch = 32
    dataset = Dataset(batch=batch)
    batch_data = dataset.train_next()
    print()