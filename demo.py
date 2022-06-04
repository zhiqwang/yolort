from yolort.v5 import load_yolov5_model
from yolort.relay import End2EndORT,End2EndTRT
import torch
import torch.nn as nn

class End2EndTensorRT(nn.Module):
    def __init__(self,weights, max_obj=100,iou_thres=0.45,score_thres=0.35,device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.yolov5 = load_yolov5_model(weights)
        self.yolov5.eval()
        self.end2end = End2EndTRT(max_obj,iou_thres,score_thres,device)
        self.end2end.eval()

    def forward(self,x):
        x = self.yolov5(x)[0]
        num_det, det_boxes, det_scores, det_classes = self.end2end(x)
        return num_det,det_boxes,det_scores,det_classes

class End2EndONNXRuntime(nn.Module):
    def __init__(self, weights, max_obj=100, iou_thres=0.45, score_thres=0.35, device=None, max_wh=640):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.yolov5 = load_yolov5_model(weights)
        self.yolov5.eval()
        self.end2end = End2EndORT(max_obj,iou_thres,score_thres,device,max_wh)
        self.end2end.eval()

    def forward(self,x):
        x = self.yolov5(x)[0]
        out = self.end2end(x)
        return out


if __name__ == '__main__':
    weights = 'yolov5s.pt'
    device = torch.device('cpu')
    ### TRT export
    # end2end = End2EndTensorRT(weights=weights,device=device)
    # end2end.eval()
    #
    # inp = torch.randn(2,3,640,640)
    # torch.onnx.export(end2end,inp,'end2end_trt.onnx',
    #                   verbose=False,training=torch.onnx.TrainingMode.EVAL,
    #                   input_names=['images'],
    #                   output_names=['num_det','det_boxes','det_scores','det_classes'],
    #                   opset_version=13,
    #                   dynamic_axes={
    #                       'images': {0: 'batch'},
    #                       'num_det': {0: 'batch'},
    #                       'det_boxes': {0: 'batch'},
    #                       'det_scores': {0: 'batch'},
    #                       'det_classes': {0: 'batch'}})

    ### ORT export
    max_wh = 640 # image size will be ok, yolov5 use 7680
    end2end = End2EndONNXRuntime(weights=weights,device=device,max_wh=max_wh)
    end2end.eval()

    inp = torch.randn(2, 3, 640, 640)

    torch.onnx.export(end2end,inp,'end2end_ort.onnx',
                      verbose=False,training=torch.onnx.TrainingMode.EVAL,
                      input_names=['images'],
                      output_names=['outputs'],
                      opset_version=13,
                      dynamic_axes={
                          'images': {0: 'batch'},
                          'outputs': {0: 'batch'}}
                      )

