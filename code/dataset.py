# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:16
# @Author wcy
import os
import pickle

import tensorflow as tf
import numpy as np
from config import TRAIN_PATH, VALID_PATH, VALID_ANSWER_PATH, TEST_A_PATH, USER_TMP_DATA_PATH
import pandas as pd
from tqdm import tqdm


class Dataset(object):

    def __init__(self, batch=10000):
        self.train_datas = pd.read_csv(TRAIN_PATH, sep="\t", chunksize=batch)

    def next(self):
        data = next(self.train_datas)
        return data


if __name__ == '__main__':
    batch = 10000
    dataset = Dataset()
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

    tmp = {name:value.shape[0] for name, value in tqdm(datas.groupby("query_id"))}
    print()