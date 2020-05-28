# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:16
# @Author wcy
import tensorflow as tf
import numpy as np
from config import TRAIN_PATH, VALID_PATH, VALID_ANSWER_PATH, TEST_A_PATH
import pandas as pd
from tqdm import tqdm


class Dataset(object):

    def __init__(self):
        self.train_datas = pd.read_csv(TRAIN_PATH, sep="\t", iterator=True)

    def next(self, batch=1000):
        data = self.train_datas.get_chunk(batch)
        return data


if __name__ == '__main__':
    # columns = ["product_id", "image_h", "image_w", "num_boxes", "query", "query_id"]
    columns = ["product_id", "query", "query_id"]
    batch = 10000
    dataset = Dataset()
    datas = pd.DataFrame(columns=columns)
    for i in tqdm(range(int(1000000/batch))):
        data = dataset.next(batch=batch)
        if data is None:
            print()
        data = data[columns]
        # res = [i for i in list(data.groupby("query_id"))]
        datas = pd.concat((datas, data))
    print()