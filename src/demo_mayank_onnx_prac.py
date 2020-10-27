import torch
import torchvision
import torch.onnx
# from models.model import create_model, load_model
from lib.model.model import create_model, load_model
# from model.model import create_model, load_model

input = torch.randn(1, 3, 512, 512)
## mobilenetv2_10
# model = create_model('mobilenetv2_10', heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=24)
## resnet18
head={'hm': 10, 'reg': 2, 'wh': 2, 'tracking': 2, 'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2}
head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256], 'dep': [256], 'rot': [256], 'dim': [256], 'amodel_offset': [256]}
model = create_model(arch='res_18', head=head, head_conv=head_conv)
# model = load_model(model, '/home/mario/Projects/Obj_det/CenterNet/exp/ctdet/face_res18/model_best.pth')

onnx_model_name="center_track.onnx"
# onnx_model_name="../onnx/model.onnx"
torch.onnx.export(model, input, onnx_model_name, input_names=["input"], output_names=["hm","wh","reg"])
# {'hm': 3, 'dep': 1, 'rot': 8, 'dim': 3, 'wh': 2, 'reg': 2}