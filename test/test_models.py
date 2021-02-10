import unittest
import torch

from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.box_head import YoloHead, PostProcess, SetCriterion

from .common_utils import TestCase

from typing import Dict


# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_model_unwrapper = {
    "PostProcess": lambda x: x[1],
}


class ModelTester(TestCase):
    def _get_strides(self):
        return [8, 16, 32]

    def _get_in_channels(self):
        return [128, 256, 512]

    def _get_anchor_grids(self):
        return [
            [ 10,  13,  16,  30,  33,  23],
            [ 30,  61,  62,  45,  59, 119],
            [116,  90, 156, 198, 373, 326],
        ]

    def _get_num_classes(self):
        return 80

    def _get_num_outputs(self):
        return self._get_num_classes() + 5

    def _get_num_anchors(self):
        return len(self._get_anchor_grids())

    def _get_anchors_shape(self):
        return [(9009, 2), (9009, 1), (9009, 2)]

    def _get_feature_shapes(self, h, w):
        strides = self._get_strides()
        in_channels = self._get_in_channels()

        return [(c, h // s, w // s) for (c, s) in zip(in_channels, strides)]

    def _get_feature_maps(self, batch_size, h, w):
        feature_shapes = self._get_feature_shapes(h, w)
        feature_maps = [torch.rand(batch_size, *f_shape) for f_shape in feature_shapes]
        return feature_maps

    def _get_head_outputs(self, batch_size, h, w):
        feature_shapes = self._get_feature_shapes(h, w)

        num_anchors = self._get_num_anchors()
        num_outputs = self._get_num_outputs()
        head_shapes = [(batch_size, num_anchors, *f_shape[1:], num_outputs) for f_shape in feature_shapes]
        head_outputs = [torch.rand(*h_shape) for h_shape in head_shapes]

        return head_outputs

    def _init_test_backbone_with_fpn(self):
        backbone_name = 'darknet_s_r3_1'
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn = darknet_pan_backbone(backbone_name, depth_multiple, width_multiple)
        return backbone_with_fpn

    def test_backbone_with_fpn(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_fpn()
        out = model(x)

        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape), (N, *out_shape[0]))
        self.assertEqual(tuple(out[1].shape), (N, *out_shape[1]))
        self.assertEqual(tuple(out[2].shape), (N, *out_shape[2]))
        self.check_jit_scriptable(model, (x,))

    def _init_test_anchor_generator(self):
        strides = self._get_strides()
        anchor_grids = self._get_anchor_grids()
        anchor_generator = AnchorGenerator(strides, anchor_grids)
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
        in_channels = self._get_in_channels()
        num_anchors = self._get_num_anchors()
        num_classes = self._get_num_classes()
        box_head = YoloHead(in_channels, num_anchors, num_classes)
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
        self.assertIsInstance(out[0]["boxes"], torch.Tensor)
        self.assertIsInstance(out[0]["labels"], torch.Tensor)
        self.assertIsInstance(out[0]["scores"], torch.Tensor)
        self.check_jit_scriptable(model, (head_outputs, anchors_tuple))

    def _init_test_criterion(self):
        weights = (1.0, 1.0, 1.0, 1.0)
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.4
        allow_low_quality_matches = True
        criterion = SetCriterion(weights, fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches)
        return criterion

    @unittest.skip("Current it isn't well implemented")
    def test_criterion(self):
        model = self._init_test_criterion()
        scripted_model = torch.jit.script(model)  # noqa


class AnchorGeneratorTester(TestCase):
    def _init_test_anchor_generator(self):
        strides = [4]
        anchor_grids = [[6, 14]]
        anchor_generator = AnchorGenerator(strides, anchor_grids)

        return anchor_generator

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 10, 10)
        features = self.get_features(images)

        model = self._init_test_anchor_generator()
        model.eval()
        anchors = model(features)

        anchor_output = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        wh_output = torch.tensor([[4.], [4.], [4.], [4.]])
        xy_output = torch.tensor([[6., 14.], [6., 14.], [6., 14.], [6., 14.]])

        self.assertEqual(len(anchors), 3)
        self.assertEqual(tuple(anchors[0].shape), (4, 2))
        self.assertEqual(tuple(anchors[1].shape), (4, 1))
        self.assertEqual(tuple(anchors[2].shape), (4, 2))
        self.assertEqual(anchors[0], anchor_output)
        self.assertEqual(anchors[1], wh_output)
        self.assertEqual(anchors[2], xy_output)


if __name__ == '__main__':
    unittest.main()
