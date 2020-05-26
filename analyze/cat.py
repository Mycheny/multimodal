# -*- coding: utf-8 -*- 
# @Time 2020/5/26 13:23
# @Author wcy
import base64
import json
import os

import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze.translator import TranslatorBaidu

train_sampleset_path = "E:/DATA/multimodal/multimodal_train_sampleset/train.sample.tsv"

testA_path = "E:/DATA/multimodal/multimodal_testA/testA.tsv"

valid_path = "E:/DATA/multimodal/multimodal_valid/valid.tsv"
valid_pics_path = "E:/DATA/multimodal/multimodal_validpics/pics"
valid_answer_path = "E:/DATA/multimodal/multimodal_valid/valid_answer.json"

labels_path = "E:/DATA/multimodal/multimodal_labels.txt"
labels = pd.read_csv(labels_path, sep="\t").values[:, 1]
baidu = TranslatorBaidu()
labels_zh = baidu.request_translate(_from="en", to="zh", text="*".join(labels)).split("*")


def change_cv2_draw(image, labels_names, boxes, sizes=16, color=(255, 0, 0)):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("SIMYOU.TTF", sizes, encoding="utf-8")
    for label_name, boxe in zip(labels_names, boxes):
        draw.text((boxe[1], boxe[0]), label_name, color, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image


if __name__ == '__main__':
    # train_sampleset = read_tsv(train_sampleset_path)

    # train_sampleset = pd.read_csv(train_sampleset_path, sep="\t")
    testA = pd.read_csv(testA_path, sep="\t")

    valid_datas = pd.read_csv(valid_path, sep="\t")
    # 各查询的商品数量
    print(valid_datas.groupby("query_id")["product_id"].count())

    valid_answer = json.load(open(valid_answer_path, "r"))
    querys = {}
    # valid_datas = valid_datas.sample(frac=1)
    for name, group in valid_datas.groupby("query_id"):
        images = {}
        cell = 110
        frames_all = np.zeros((cell * 6, cell * 6, 3), np.uint8)
        product_ids = list(group["product_id"])
        # query_list = [key for key, value in list(valid_datas.groupby("query"))]
        # query_zhs = baidu.request_translate(_from="en", to="zh", text="*".join(query_list)).split("*")
        # querys = {en:zh for zh, en in zip(query_zhs, query_list)}
        for index, tup in enumerate(group.itertuples()):
            product_id = tup.product_id
            image_path = os.path.join(valid_pics_path, f"{product_id}.jpg")
            if not os.path.exists(image_path):
                continue
            image_h = tup.image_h
            image_w = tup.image_w
            num_boxes = tup.num_boxes
            boxes = np.frombuffer(base64.b64decode(tup.boxes), dtype=np.float32).reshape(num_boxes, 4)
            features = np.frombuffer(base64.b64decode(tup.features), dtype=np.float32).reshape(num_boxes, 2048)
            class_labels = np.frombuffer(base64.b64decode(tup.class_labels), dtype=np.int64).reshape(num_boxes)
            class_labels_names = [labels[index] for index in class_labels]
            class_labels_names_zh = [labels_zh[index] for index in class_labels]
            query = tup.query
            if query in querys.keys():
                query_zh = querys[query]
            else:
                query_zh = baidu.request_translate(_from="en", to="zh", text=query)
                querys[query] = query_zh
            query_id = tup.query_id
            image = cv2.imread(image_path)
            frames_all[(index // 6) * cell:(index // 6 + 1) * cell, (index % 6) * cell:(index % 6 + 1) * cell, :] = \
            cv2.resize(image, (cell, cell))
            for boxe, text in zip(boxes, class_labels_names_zh):
                cv2.rectangle(image, (boxe[1], boxe[0]), (boxe[3], boxe[2]), (0, 255, 0))
                image = change_cv2_draw(image, class_labels_names_zh, boxes)
            images[product_id] = cv2.resize(image, (cell * 5, cell * 5))
            # print(query, query_zh)
            # cv2.imshow("image", image)
            # cv2.waitKey(1)
        print(query_zh)
        frames = np.zeros((cell * 6, cell * 6, 3), np.uint8)
        for i, product_id in enumerate(valid_answer[str(name)]):
            if product_id in images.keys():
                image = images[product_id]
                image = cv2.resize(image, (cell, cell))
            else:
                image = np.ones((cell, cell, 3), dtype=np.uint8) * 255
            frames[(i // 6) * cell:(i // 6 + 1) * cell, (i % 6) * cell:(i % 6 + 1) * cell, :] = image


        def event_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                xy = "%d,%d" % (x, y)
                id = x//cell+y//cell*6
                product_id = product_ids[id]
                if product_id in images.keys():
                    image = images[product_id]
                    frames[cell:cell+cell*5, :cell*5, :] = image


        winName = "frames_all"
        cv2.namedWindow(winName)
        cv2.setMouseCallback(winName, event_mouse)
        while True:
            cv2.imshow(winName, frames_all)
            cv2.imshow("frames", frames)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
