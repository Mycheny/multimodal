# -*- coding: utf-8 -*- 
# @Time 2020/5/27 14:17
# @Author wcy
import numpy as np
import tensorflow as tf

from triplet import Triplet


class Model(Triplet):

    def __init__(self):
        super().__init__()

    def build(self):
        dim = 128
        batch = 5
        labels = np.random.randint(0, 5, (batch,))
        embeddings = (np.round(np.random.random((batch, dim)), 1)).astype(np.float32)
        margin = 0.5
        all_triplet_loss = self.batch_all_triplet_loss(labels, embeddings, margin)
        hard_triplet_loss = self.batch_hard_triplet_loss(labels, embeddings, margin)


if __name__ == '__main__':
    pass