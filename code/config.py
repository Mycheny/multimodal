# -*- coding: utf-8 -*- 
# @Time 2020/5/28 10:54
# @Author wcy
import os

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