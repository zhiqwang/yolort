"""
detect from U版 yolov5
"""

import os
import torch
#from yolort.models import YOLO
from yolort.models import YOLOv5
from yolort.relaying.trace_wrapper import get_trace_module

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

score_thresh = 0.35
iou = 0.45
checkpoint_path = "./model_file/best_0128.pt"



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# YOLO.load_from_yolov5之后后处理nms，无前处理
# YOLOv5.load_from_yolov5已做前处理，并与U版做了精度对齐

model_yolort = YOLOv5.load_from_yolov5(
    checkpoint_path,
    score_thresh=score_thresh
)

model_yolort = model_yolort.to(device)
model_yolort = model_yolort.eval()

traced_model = get_trace_module(model_yolort, input_shape=(576, 2016))

traced_model.save("best_traced_nopre.pt")
