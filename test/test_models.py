import unittest
import torch

from models.backbone import darknet
from models.anchor_utils import AnchorGenerator
from models.box_head import YoloHead, PostProcess, SetCriterion

from .common_utils import TestCase


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

    def _init_test_backbone(self):
        backbone = darknet()
        return backbone

    def test_yolo_backbone_script(self):
        model, _ = self._init_test_backbone()
        N, H, W = 8, 416, 352
        x = torch.rand(N, 3, H, W)
        out = model(x)
        strides = self._get_strides()
        grid_sizes = [(H // s, W // s) for s in strides]
        self.assertEqual(len(out), 3)
        self.assertEqual(tuple(out[0].shape[-2:]), grid_sizes[0])
        self.assertEqual(tuple(out[1].shape[-2:]), grid_sizes[1])
        self.assertEqual(tuple(out[2].shape[-2:]), grid_sizes[2])
        self.check_jit_scriptable(model, (x,))

    def _init_test_anchor_generator(self):
        strides = self._get_strides()
        anchor_grids = self._get_anchor_grids()
        anchor_generator = AnchorGenerator(strides, anchor_grids)
        return anchor_generator

    def test_anchor_generator_script(self):
        model = self._init_test_anchor_generator()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_yolo_head(self):
        in_channels = self._get_in_channels()
        num_anchors = 3
        num_classes = 80
        box_head = YoloHead(in_channels, num_anchors, num_classes)
        return box_head

    def test_yolo_head_script(self):
        model = self._init_test_yolo_head()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_postprocessors(self):
        score_thresh = 0.5
        nms_thresh = 0.45
        detections_per_img = 100
        postprocessors = PostProcess(score_thresh, nms_thresh, detections_per_img)
        return postprocessors

    def test_postprocessors_script(self):
        model = self._init_test_postprocessors()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_criterion(self):
        weights = (1.0, 1.0, 1.0, 1.0)
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.4
        allow_low_quality_matches = True
        criterion = SetCriterion(weights, fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches)
        return criterion

    @unittest.skip("Current it isn't well implemented")
    def test_criterion_script(self):
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
