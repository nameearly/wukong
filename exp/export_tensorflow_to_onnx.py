import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

from model.tensorflow.wukong import Wukong
from exp.criteo_kaggle_constants import (
    NUM_CAT_FEATURES,
    NUM_DENSE_FEATURES,
    NUM_SPARSE_EMBS,
)

BATCH_SIZE = 2
# NOTE: 需要与训练时使用的、数据集特定的词表大小保持一致。
# 否则导出模型的嵌入表形状会与训练时保存的权重悄悄产生偏差，
# 导致权重无法加载或导出的模型没有实际意义。
DIM_EMB = 128
assert NUM_CAT_FEATURES == len(NUM_SPARSE_EMBS)


model = Wukong(
    num_layers=2,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_emb=DIM_EMB,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    num_emb_lcb=16,
    num_emb_fmb=23,
    rank_fmb=8,
    num_hidden_wukong=2,
    dim_hidden_wukong=16,
    num_hidden_head=2,
    dim_hidden_head=32,
    dim_output=1,
    dropout=0.5,
)

sparse_inputs = tf.constant(
    np.column_stack(
        [
            np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
            for i in range(NUM_CAT_FEATURES)
        ]
    ).astype(np.int32)
)
dense_inputs = tf.constant(
    np.random.rand(BATCH_SIZE, NUM_DENSE_FEATURES).astype(np.float32)
)
outputs = model((sparse_inputs, dense_inputs))
print("Model output shape:", outputs.shape)
input_signature = [
    tf.TensorSpec(
        shape=sparse_inputs.shape, dtype=tf.int32, name="sparse_inputs"
    ),  # sparse_inputs: must be integer dtype for embedding lookup
    tf.TensorSpec(
        shape=dense_inputs.shape, dtype=tf.float32, name="dense_inputs"
    ),  # dense_inputs
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model, input_signature=input_signature, opset=17
)
# 保存ONNX模型
onnx.save(onnx_model, "wukong_tensorflow_model.onnx")
