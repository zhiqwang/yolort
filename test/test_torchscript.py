import unittest

import torch

from models.backbone import darknet
from models.anchor_utils import AnchorGenerator
from models.box_head import YoloHead, PostProcess, SetCriterion
from models import yolov5s, yolov5m, yolov5l


class ModelTester(unittest.TestCase):

    def _init_test_backbone(self):
        backbone = darknet()
        return backbone

    def test_yolo_backbone_script(self):
        model, _ = self._init_test_backbone()
        torch.jit.script(model)

    def _init_test_anchor_generator(self):
        strides = [8, 16, 32]
        anchor_grids = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        anchor_generator = AnchorGenerator(strides, anchor_grids)
        return anchor_generator

    def test_anchor_generator_script(self):
        model = self._init_test_anchor_generator()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_yolo_head(self):
        in_channels = [128, 256, 512]
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

    def test_yolov5s_script(self):
        model = yolov5s()
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)[1]
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))

    def test_yolov5m_script(self):
        model = yolov5m()
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)[1]
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))

    def test_yolov5l_script(self):
        model = yolov5l()
        model.eval()

        scripted_model = torch.jit.script(model)
        scripted_model.eval()

        x = [torch.rand(3, 416, 320), torch.rand(3, 480, 352)]

        out = model(x)
        out_script = scripted_model(x)[1]
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))


if __name__ == "__main__":
    unittest.main()
