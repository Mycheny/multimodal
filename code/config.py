# -*- coding: utf-8 -*- 
# @Time 2020/5/28 10:54
# @Author wcy
import os
import platform
system = platform.system()

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
# 外部资源目录
EXTERNAL_RESOURCE_PATH = os.path.join(BASE_DIR, 'external_resources')
# 用户数据目录
USER_DATA_PATH = os.path.join(BASE_DIR, 'user_data')
# 用户模型路径
USER_MODEL_DATA_PATH = os.path.join(USER_DATA_PATH, 'model_data')
# 用户临时文件路径
USER_TMP_DATA_PATH = os.path.join(USER_DATA_PATH, 'tmp_data')
# bert模型路径
BERT_PRE_MODEL_PATH = os.path.join(BASE_DIR, EXTERNAL_RESOURCE_PATH, 'bert_pre_model/multi_cased_L-12_H-768_A-12')

# 数据路径
if system=="Liunx":
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    TRAIN_PATH = os.path.join(DATA_PATH, 'train', f"train.tsv")
    VALID_PATH = os.path.join(DATA_PATH, 'valid', f"valid.tsv")
    VALID_ANSWER_PATH = os.path.join(DATA_PATH, 'valid', f"valid_answer.json")
    TEST_A_PATH = os.path.join(DATA_PATH, 'testA', f"testA.tsv")
    TEST_B_PATH = os.path.join(DATA_PATH, 'testB', f"testB.tsv")
else:
    DATA_PATH = "E:\\DATA\\multimodal"
    TRAIN_PATH = os.path.join(DATA_PATH, 'multimodal_train', f"train.tsv")
    VALID_PATH = os.path.join(DATA_PATH, 'multimodal_valid', f"valid.tsv")
    VALID_ANSWER_PATH = os.path.join(DATA_PATH, 'multimodal_valid', f"valid_answer.json")
    TEST_A_PATH = os.path.join(DATA_PATH, 'multimodal_testA', f"testA.tsv")
    TEST_B_PATH = os.path.join(DATA_PATH, 'multimodal_testB', f"testB.tsv")