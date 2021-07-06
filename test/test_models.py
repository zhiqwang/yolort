# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import Tensor

from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.transformer import darknet_tan_backbone
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.box_head import YOLOHead, PostProcess, SetCriterion

from .common_utils import TestCase

from typing import Dict


# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_model_unwrapper = {
    "PostProcess": lambda x: x[1],
}


class ModelTester(TestCase):
    strides = [8, 16, 32]
    in_channels = [128, 256, 512]
    anchor_grids = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326],
    ]
    num_classes = 80
    num_outputs = num_classes + 5
    num_anchors = len(anchor_grids)

    def _get_feature_shapes(self, h, w):
        strides = self.strides
        in_channels = self.in_channels

        return [(c, h // s, w // s) for (c, s) in zip(in_channels, strides)]

    def _get_feature_maps(self, batch_size, h, w):
        feature_shapes = self._get_feature_shapes(h, w)
        feature_maps = [torch.rand(batch_size, *f_shape) for f_shape in feature_shapes]
        return feature_maps

    def _get_head_outputs(self, batch_size, h, w):
        feature_shapes = self._get_feature_shapes(h, w)

        num_anchors = self.num_anchors
        num_outputs = self.num_outputs
        head_shapes = [(batch_size, num_anchors, *f_shape[1:], num_outputs) for f_shape in feature_shapes]
        head_outputs = [torch.rand(*h_shape) for h_shape in head_shapes]

        return head_outputs

    def _init_test_backbone_with_pan_r3_1(self):
        backbone_name = 'darknet_s_r3_1'
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple)
        return backbone_with_fpn

    def test_backbone_with_pan_r3_1(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_r3_1()
        out = model(x)

        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), (N, *out_shape[0]))
        self.assertEqual(tuple(out[1].shape), (N, *out_shape[1]))
        self.assertEqual(tuple(out[2].shape), (N, *out_shape[2]))
        self.check_jit_scriptable(model, (x,))

    def _init_test_backbone_with_pan_r4_0(self):
        backbone_name = 'darknet_s_r4_0'
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple)
        return backbone_with_fpn

    def test_backbone_with_pan_r4_0(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_r4_0()
        out = model(x)

        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), (N, *out_shape[0]))
        self.assertEqual(tuple(out[1].shape), (N, *out_shape[1]))
        self.assertEqual(tuple(out[2].shape), (N, *out_shape[2]))
        self.check_jit_scriptable(model, (x,))

    def _init_test_backbone_with_pan_tr(self):
        backbone_name = 'darknet_s_r4_0'
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn_tr = darknet_tan_backbone(backbone_name, depth_multiple, width_multiple)
        return backbone_with_fpn_tr

    def test_backbone_with_pan_tr(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_tr()
        out = model(x)

        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), (N, *out_shape[0]))
        self.assertEqual(tuple(out[1].shape), (N, *out_shape[1]))
        self.assertEqual(tuple(out[2].shape), (N, *out_shape[2]))
        self.check_jit_scriptable(model, (x,))

    def _init_test_anchor_generator(self):
        anchor_generator = AnchorGenerator(self.strides, self.anchor_grids)
        return anchor_generator

    def test_anchor_generator(self):
        N, H, W = 4, 416, 352
        feature_maps = self._get_feature_maps(N, H, W)
        model = self._init_test_anchor_generator()
        anchors = model(feature_maps)

        self.assertEqual(len(anchors), 3)
        self.assertEqual(tuple(anchors[0].shape), (9009, 2))
        self.assertEqual(tuple(anchors[1].shape), (9009, 1))
        self.assertEqual(tuple(anchors[2].shape), (9009, 2))
        self.check_jit_scriptable(model, (feature_maps,))

    def _init_test_yolo_head(self):
        box_head = YOLOHead(self.in_channels, self.num_anchors, self.strides, self.num_classes)
        return box_head

    def test_yolo_head(self):
        N, H, W = 4, 416, 352
        feature_maps = self._get_feature_maps(N, H, W)
        model = self._init_test_yolo_head()
        head_outputs = model(feature_maps)
        self.assertEqual(len(head_outputs), 3)

        target_head_outputs = self._get_head_outputs(N, H, W)

        self.assertEqual(head_outputs[0].shape, target_head_outputs[0].shape)
        self.assertEqual(head_outputs[1].shape, target_head_outputs[1].shape)
        self.assertEqual(head_outputs[2].shape, target_head_outputs[2].shape)
        self.check_jit_scriptable(model, (feature_maps,))

    def _init_test_postprocessors(self):
        score_thresh = 0.5
        nms_thresh = 0.45
        detections_per_img = 100
        postprocessors = PostProcess(score_thresh, nms_thresh, detections_per_img)
        return postprocessors

    def test_postprocessors(self):
        N, H, W = 4, 416, 352
        feature_maps = self._get_feature_maps(N, H, W)
        head_outputs = self._get_head_outputs(N, H, W)

        anchor_generator = self._init_test_anchor_generator()
        anchors_tuple = anchor_generator(feature_maps)
        model = self._init_test_postprocessors()
        out = model(head_outputs, anchors_tuple)

        self.assertEqual(len(out), N)
        self.assertIsInstance(out[0], Dict)
        self.assertIsInstance(out[0]["boxes"], Tensor)
        self.assertIsInstance(out[0]["labels"], Tensor)
        self.assertIsInstance(out[0]["scores"], Tensor)
        self.check_jit_scriptable(model, (head_outputs, anchors_tuple))

    def test_criterion(self):
        N, H, W = 4, 640, 640
        head_outputs = self._get_head_outputs(N, H, W)
        targets = torch.tensor([
            [0.0000, 7.0000, 0.0714, 0.3749, 0.0760, 0.0654],
            [0.0000, 1.0000, 0.1027, 0.4402, 0.2053, 0.1920],
            [1.0000, 5.0000, 0.4720, 0.6720, 0.3280, 0.1760],
            [3.0000, 3.0000, 0.6305, 0.3290, 0.3274, 0.2270],
        ])
        loss_calculator = SetCriterion(self.strides, self.anchor_grids)
        out = loss_calculator(targets, head_outputs)
        self.assertIsInstance(out, Dict)
        self.assertIsInstance(out['cls_logits'], Tensor)
        self.assertIsInstance(out['bbox_regression'], Tensor)
        self.assertIsInstance(out['objectness'], Tensor)
