import torch

from model.pytorch.wukong import Wukong
from exp.criteo_kaggle_constants import (
    NUM_CAT_FEATURES,
    NUM_DENSE_FEATURES,
    NUM_SPARSE_EMBS,
)

BATCH_SIZE = 1
# NOTE: 需要与训练时使用的、数据集特定的词表大小保持一致。
# 否则导出模型的嵌入表形状会与训练时保存的权重不一致（形状不匹配），
# 导出的 ONNX 也就无法代表真实训练得到的模型。
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
model.eval()
dummy_sparse_input = torch.stack(
    [
        torch.randint(0, NUM_SPARSE_EMBS[i], (BATCH_SIZE,), dtype=torch.long)
        for i in range(NUM_CAT_FEATURES)
    ],
    dim=1,
)
dummy_dense_input = torch.randn(BATCH_SIZE, NUM_DENSE_FEATURES)
output = model(dummy_sparse_input, dummy_dense_input)
print("Output shape:", output.shape)
torch.onnx.export(
    model,
    (dummy_sparse_input, dummy_dense_input),
    "wukong_model.onnx",
    input_names=["sparse_input", "dense_input"],
    output_names=["output"],
    opset_version=17,
)
