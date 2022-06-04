import numpy as np
import torch
import random
from torch import nn,Tensor



class TRTNMS(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                boxes: Tensor, # B,anchors,4
                scores: Tensor, # B,anchors,80
                background_class: int = -1,
                box_coding: int = 0,
                iou_threshold: float = 0.45,
                max_output_boxes: int = 100,
                plugin_version: str = '1',
                score_activation: int = 0,
                score_threshold: float = 0.35):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0,max_output_boxes,(batch_size,1))
        det_boxes = torch.randn(batch_size,max_output_boxes,4)
        det_scores = torch.randn(batch_size,max_output_boxes)
        det_classes = torch.randint(0,num_classes,(batch_size,max_output_boxes))

        return num_det,det_boxes,det_scores,det_classes

    @staticmethod
    def symbolic(g,
                boxes: Tensor, # B,anchors,4
                scores: Tensor, # B,anchors,80
                background_class: int = -1,
                box_coding: int = 0,
                iou_threshold: float = 0.45,
                max_output_boxes: int = 100,
                plugin_version: str = '1',
                score_activation: int = 0,
                score_threshold: float = 0.35):

        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4)



class ORTNMS(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                nmsboxes: Tensor,  # B,anchors,4 # have been multiplied max_wh
                scores: Tensor,  # B,1,anchors # only one class score
                max_output_boxes_per_class: Tensor = torch.tensor([100]),
                iou_threshold: Tensor = torch.tensor([0.45]),
                score_threshold: Tensor = torch.tensor([0.35])):
        batch, _, anchors = scores.shape
        num_det = random.randint(0,100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0]
        idxs = torch.arange(100, 100 + num_det)
        zeros = torch.zeros((num_det,), dtype = torch.int64)
        selected_indices = torch.cat([batches[None],zeros[None],idxs[None]],0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g,
                 nmsboxes: Tensor,  # B,anchors,4
                 scores: Tensor,  # B,1,anchors
                 max_output_boxes_per_class: Tensor, # [1]
                 iou_threshold: Tensor, # [1]
                 score_threshold: Tensor): # [1]
        return g.op(
            'NonMaxSuppression',
            nmsboxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)

class End2EndORT(nn.Module):
    def __init__(self,max_obj=100,iou_thres=0.45,score_thres=0.35, device=None, max_wh=640):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32, device=self.device)

    def forward(self, x):  # b,18900,85
        box = x[:, :, :4] # b,18900,4
        conf = x[:, :, 4:5] # b,18900,1
        score = x[:, :, 5:] # b,18900,80
        score *= conf # b,18900,80
        box @= self.convert_matrix # b,18900,4
        objScore, objCls = score.max(2, keepdim=True) # b,18900,1   # b,18900,1
        dis = objCls.float() * self.max_wh # b,18900,4
        nmsbox = box + dis # b,18900,4
        objScore1 = objScore.transpose(1, 2).contiguous()
        selected_indices = ORTNMS.apply(nmsbox,objScore1,
                                  self.max_obj,
                                  self.iou_threshold,
                                  self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        resBoxes = box[X, Y, :]
        resClasses = objCls[X, Y, :]
        resScores = objScore[X, Y, :]
        X = X.unsqueeze(1)
        X = X.float()
        resClasses = resClasses.float()
        out = torch.concat([X, resBoxes, resClasses, resScores], 1)
        return out

class End2EndTRT(nn.Module):
    def __init__(self,max_obj=100,iou_thres=0.45,score_thres=0.35, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')

        self.background_class = -1,
        self.box_coding = 0,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,device=self.device)

    def forward(self,x): # b,18900,85
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        num_det,det_boxes,det_scores,det_classes = TRTNMS.apply(box,score,
                                                                self.background_class,
                                                                self.box_coding,
                                                                self.iou_threshold,
                                                                self.max_obj,
                                                                self.plugin_version,
                                                                self.score_activation,
                                                                self.score_threshold)
        return num_det,det_boxes,det_scores,det_classes

def main_ort(pred,device):
    end2end = End2EndORT(device=device)
    end2end.eval()
    torch.onnx.export(end2end, pred, 'ort_nms.onnx', opset_version=11)

def main_trt(pred,device):
    end2end = End2EndTRT(device=device)
    end2end.eval()
    torch.onnx.export(end2end, pred, 'trt_nms.onnx', opset_version=11)



if __name__ == '__main__':
    pred = np.load('../../pred.npy')
    pred = torch.from_numpy(pred)

    device = torch.device('cpu')
    main_ort(pred,device)
    main_trt(pred,device)


