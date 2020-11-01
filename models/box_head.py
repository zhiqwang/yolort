# Modified from ultralytics/yolov5 by Zhiqiang Wang
from typing import List, Dict, Optional
import torch
from torch import nn, Tensor
from torchvision.ops.boxes import batched_nms

from utils.general import box_cxcywh_to_xyxy


class YoloHead(nn.Module):
    def __init__(self, nc=80, stride=[8., 16., 32.], anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.stride = stride

    def get_result_from_m(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.m[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.m:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.m:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        # x = x.copy()  # for profiling
        device = x[0].device
        z = torch.jit.annotate(List[Tensor], [])  # inference output
        for i in range(self.nl):
            x[i] = self.get_result_from_m(x[i], i)  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if not isinstance(self.stride, Tensor):
                    self.stride = torch.tensor(self.stride, device=device)

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        prediction = torch.cat(z, 1)

        return prediction

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class PostProcess(nn.Module):
    """Performs Non-Maximum Suppression (NMS) on inference results"""
    def __init__(
        self,
        conf_thres: float,
        iou_thres: float,
        merge: bool = False,
        detections_per_img: int = 300,
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.merge = merge
        self.detections_per_img = detections_per_img  # maximum number of detections per image

    def forward(
        self,
        prediction: Tensor,
        target_sizes: Optional[Tensor] = None,
    ) -> List[Dict[str, Tensor]]:
        results = torch.jit.annotate(List[Dict[str, Tensor]], [])

        for pred in prediction:  # image index, image inference
            # Compute conf
            scores = pred[:, 5:] * pred[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            boxes = box_cxcywh_to_xyxy(pred[:, :4])

            # remove low scoring boxes
            inds, labels = torch.where(scores > self.conf_thres)
            boxes, scores = boxes[inds], scores[inds, labels]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, labels, self.iou_thres)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return results
