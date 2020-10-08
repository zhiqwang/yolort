# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
from typing import List
import torch
from torch import nn, Tensor

from torchvision.ops import nms, box_iou


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
        z: List[Tensor] = []  # inference output
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

        return torch.cat(z, 1)

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
        agnostic: bool = False,
    ):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.merge = merge
        self.agnostic = agnostic

    def forward(self, prediction: Tensor):
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > self.conf_thres  # candidates

        # Settings
        _, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        redundant = True  # require redundant detections
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

        output = torch.jit.annotate(List[Tensor], [])
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = box_cxcywh_to_xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                inds = torch.nonzero(x[:, 5:] > self.conf_thres)
                i = inds[:, 0]
                j = inds[:, 1]
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_thres]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5:6] * (0 if self.agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, self.iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if self.merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                iou = box_iou(boxes[i], boxes) > self.iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output.append(x[i])

        return torch.stack(output, dim=0)


def box_cxcywh_to_xyxy(x):
    """ Convert BoxMode of boxes from XYWHA_REL to XYXY_REL.
    Args:
        boxes (Tensor): XYWHA_REL BoxMode
            default BoxMode from priorbox generator layers.
    Return:
        boxes (Tensor): XYXY_REL BoxMode
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
