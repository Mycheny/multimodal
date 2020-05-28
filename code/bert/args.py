import os
import tensorflow as tf
from config import BERT_PRE_MODEL_PATH, USER_TMP_DATA_PATH

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)

# model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12/')
model_dir = BERT_PRE_MODEL_PATH
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(model_dir, '../temp')
vocab_file = os.path.join(model_dir, 'vocab.txt')

num_train_epochs = 35
batch_size = 64
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = None

# pb名字
pb_file = os.path.join(USER_TMP_DATA_PATH, 'bert.pb')