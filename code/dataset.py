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

    def __init__(self, batch=10000, max_boxes_num=10):
        self.max_boxes_num = max_boxes_num
        self.batch = batch
        self.text2vec = BertEncode(graph_path=None)
        self.labels_dict = pd.read_csv(LABELS_NAMES_PATH, sep="\t").values[:, 1]
        self.labels_vector_dict = self.text2vec.encode(self.labels_dict)
        self.train_datas = pd.read_csv(TRAIN_PATH, sep="\t", chunksize=self.batch)

    def deal_data(self, data, index):
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
        if len(query) < 512:
            query_vector = self.text2vec.encode(query)
        else:
            query_vector = [self.text2vec.encode(q)[0] for q in tqdm(query)]
        query_id = data.query_id.values
        return {"product_id": product_id,
                "image_h": image_h,
                "image_w": image_w,
                "num_boxes": num_boxes,
                "boxes": boxes,
                "features": features,
                "class_labels": class_labels,
                "class_labels_names": class_labels_names,
                "class_labels_vector": class_labels_vector,
                "query": query,
                "query_vector": query_vector,
                "query_id": query_id}

    def train_next(self, index=0):
        while True:
            try:
                data = next(self.train_datas)
                data = self.deal_data(data, index)
                query_vector = data["query_vector"]

                boxe_vector = self.build_boxe_vector(data["image_h"], data["image_w"], data["boxes"])
                boxes_mask = self.build_mask(data["num_boxes"])
                boxe_vector = self.expand_zeros(boxe_vector)
                feature_vector = self.expand_zeros(data["features"])
                label_vector = self.expand_zeros(data["class_labels_vector"])
                triplet_labels = self.build_triplet_labels(data["query_id"], data["query"])
                return {
                    "query_vector": query_vector,
                    "boxes_mask": boxes_mask,
                    "boxe_vector": boxe_vector,
                    "feature_vector": feature_vector,
                    "label_vector": label_vector,
                    "triplet_labels": triplet_labels
                }
            except StopIteration as e:
                print(e)
                self.train_datas = pd.read_csv(TRAIN_PATH, sep="\t", chunksize=self.batch)
            except Exception as e:
                print(e)

    def build_boxe_vector(self, image_h, image_w, boxes):
        areas = image_h * image_w
        areas_ratio = [
            np.array([[(x2 - x1) / 2 / w, (y2 - y1) / 2 / h, (x2 - x1) * (y2 - y1) / area] for y1, x1, y2, x2 in boxe])
            for boxe, area, h, w in zip(boxes, areas, image_h, image_w)]
        return areas_ratio

    def build_triplet_labels(self, query_id, query):
        unrepeat_query_id, unrepeat_query = set(query_id), set(query)
        unrepeat_query_id = list(unrepeat_query_id)
        triplet_labels = np.array([unrepeat_query_id.index(id) for id in query_id])
        triplet_labels = np.concatenate((triplet_labels, triplet_labels))
        # assert len(unrepeat_query_id)==len(unrepeat_query), 'query_id != query'
        # triplet_labels = np.concatenate((np.arange(0, len(query_id)), np.arange(0, len(query_id))))
        return triplet_labels

    def expand_zeros(self, params):
        res = np.array([np.pad(param, ((0, self.max_boxes_num - len(param)), (0, 0)), mode='constant') if len(
            param) < self.max_boxes_num else param[:self.max_boxes_num] for param in params])
        return res

    def build_mask(self, param):
        masks = np.zeros((len(param), self.max_boxes_num))
        for i, p in enumerate(param):
            masks[i, :p] = 1
        return masks


def analyze():
    batch = 10000
    dataset = Dataset(batch=batch)
    pkl_file_path = os.path.join(USER_TMP_DATA_PATH, "train.pkl")
    if not os.path.exists(pkl_file_path):
        # columns = ["product_id", "image_h", "image_w", "num_boxes", "query", "query_id"]
        columns = ["product_id", "num_boxes", "query", "query_id"]
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
    # analyze()
    batch = 300
    dataset = Dataset(batch=batch)
    i = 0
    while True:
        batch_data = dataset.train_next(index=i)
        if i % 10 == 0: print(i)
        i += 1
