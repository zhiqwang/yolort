import contextlib
import io
import os
import warnings
from pathlib import Path

import pytest
import torch
from torch import Tensor
from yolort import models
from yolort.models import YOLOv5
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import YOLOHead, PostProcess, SetCriterion
from yolort.models.transformer import darknet_tan_backbone


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


def _check_jit_scriptable(nn_module, args, unwrapper=None, skip=False):
    """
    Check that a nn.Module's results in TorchScript match eager and that it can be exported
    https://github.com/pytorch/vision/blob/12fd3a6/test/test_models.py#L90-L141
    """

    def assert_export_import_module(m, args):
        """
        Check that the results of a model are the same after saving and loading
        """

        def get_export_import_copy(m):
            """
            Save and load a TorchScript model
            """
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            imported = torch.jit.load(buffer)
            return imported

        m_import = get_export_import_copy(m)
        with freeze_rng_state():
            results = m(*args)
        with freeze_rng_state():
            results_from_imported = m_import(*args)
        tol = 3e-4
        try:
            torch.testing.assert_close(
                results, results_from_imported, atol=tol, rtol=tol
            )
        except ValueError:
            # custom check for the models that return named tuples:
            # we compare field by field while ignoring None as assert_close can't handle None
            for a, b in zip(results, results_from_imported):
                if a is not None:
                    torch.testing.assert_close(a, b, atol=tol, rtol=tol)

    TEST_WITH_SLOW = os.getenv("PYTORCH_TEST_WITH_SLOW", "0") == "1"
    if not TEST_WITH_SLOW or skip:
        # TorchScript is not enabled, skip these tests
        msg = (
            f"The check_jit_scriptable test for {nn_module.__class__.__name__} "
            "was skipped. This test checks if the module's results in TorchScript "
            "match eager and that it can be exported. To run these tests make "
            "sure you set the environment variable PYTORCH_TEST_WITH_SLOW=1 and "
            "that the test is not manually skipped."
        )
        warnings.warn(msg, RuntimeWarning)
        return None

    sm = torch.jit.script(nn_module)

    with freeze_rng_state():
        eager_out = nn_module(*args)

    with freeze_rng_state():
        script_out = sm(*args)
        if unwrapper:
            script_out = unwrapper(script_out)

    torch.testing.assert_close(eager_out, script_out, atol=1e-4, rtol=1e-4)
    assert_export_import_module(sm, args)


class TestModel:
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
        head_shapes = [
            (batch_size, num_anchors, *f_shape[1:], num_outputs)
            for f_shape in feature_shapes
        ]
        head_outputs = [torch.rand(*h_shape) for h_shape in head_shapes]

        return head_outputs

    def _init_test_backbone_with_pan_r3_1(self):
        backbone_name = "darknet_s_r3_1"
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn = darknet_pan_backbone(
            backbone_name, depth_multiple, width_multiple
        )
        return backbone_with_fpn

    def test_backbone_with_pan_r3_1(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_r3_1()
        out = model(x)

        assert len(out) == 3
        assert tuple(out[0].shape) == (N, *out_shape[0])
        assert tuple(out[1].shape) == (N, *out_shape[1])
        assert tuple(out[2].shape) == (N, *out_shape[2])
        _check_jit_scriptable(model, (x,))

    def _init_test_backbone_with_pan_r4_0(self):
        backbone_name = "darknet_s_r4_0"
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn = darknet_pan_backbone(
            backbone_name, depth_multiple, width_multiple
        )
        return backbone_with_fpn

    def test_backbone_with_pan_r4_0(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_r4_0()
        out = model(x)

        assert len(out) == 3
        assert tuple(out[0].shape) == (N, *out_shape[0])
        assert tuple(out[1].shape) == (N, *out_shape[1])
        assert tuple(out[2].shape) == (N, *out_shape[2])
        _check_jit_scriptable(model, (x,))

    def _init_test_backbone_with_pan_tr(self):
        backbone_name = "darknet_s_r4_0"
        depth_multiple = 0.33
        width_multiple = 0.5
        backbone_with_fpn_tr = darknet_tan_backbone(
            backbone_name, depth_multiple, width_multiple
        )
        return backbone_with_fpn_tr

    def test_backbone_with_pan_tr(self):
        N, H, W = 4, 416, 352
        out_shape = self._get_feature_shapes(H, W)

        x = torch.rand(N, 3, H, W)
        model = self._init_test_backbone_with_pan_tr()
        out = model(x)

        assert len(out) == 3
        assert tuple(out[0].shape) == (N, *out_shape[0])
        assert tuple(out[1].shape) == (N, *out_shape[1])
        assert tuple(out[2].shape) == (N, *out_shape[2])
        _check_jit_scriptable(model, (x,))

    def _init_test_anchor_generator(self):
        anchor_generator = AnchorGenerator(self.strides, self.anchor_grids)
        return anchor_generator

    def test_anchor_generator(self):
        N, H, W = 4, 416, 352
        feature_maps = self._get_feature_maps(N, H, W)
        model = self._init_test_anchor_generator()
        anchors = model(feature_maps)

        assert len(anchors) == 3
        assert tuple(anchors[0].shape) == (9009, 2)
        assert tuple(anchors[1].shape) == (9009, 1)
        assert tuple(anchors[2].shape) == (9009, 2)
        _check_jit_scriptable(model, (feature_maps,))

    def _init_test_yolo_head(self):
        box_head = YOLOHead(
            self.in_channels, self.num_anchors, self.strides, self.num_classes
        )
        return box_head

    def test_yolo_head(self):
        N, H, W = 4, 416, 352
        feature_maps = self._get_feature_maps(N, H, W)
        model = self._init_test_yolo_head()
        head_outputs = model(feature_maps)
        assert len(head_outputs) == 3

        target_head_outputs = self._get_head_outputs(N, H, W)

        assert head_outputs[0].shape == target_head_outputs[0].shape
        assert head_outputs[1].shape == target_head_outputs[1].shape
        assert head_outputs[2].shape == target_head_outputs[2].shape
        _check_jit_scriptable(model, (feature_maps,))

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

        assert len(out) == N
        assert isinstance(out[0], dict)
        assert isinstance(out[0]["boxes"], Tensor)
        assert isinstance(out[0]["labels"], Tensor)
        assert isinstance(out[0]["scores"], Tensor)
        _check_jit_scriptable(model, (head_outputs, anchors_tuple))

    def test_criterion(self):
        N, H, W = 4, 640, 640
        head_outputs = self._get_head_outputs(N, H, W)

        targets = torch.tensor(
            [
                [0.0000, 7.0000, 0.0714, 0.3749, 0.0760, 0.0654],
                [0.0000, 1.0000, 0.1027, 0.4402, 0.2053, 0.1920],
                [1.0000, 5.0000, 0.4720, 0.6720, 0.3280, 0.1760],
                [3.0000, 3.0000, 0.6305, 0.3290, 0.3274, 0.2270],
            ]
        )
        criterion = SetCriterion(
            self.num_anchors, self.strides, self.anchor_grids, self.num_classes
        )
        losses = criterion(targets, head_outputs)
        assert isinstance(losses, dict)
        assert isinstance(losses["cls_logits"], Tensor)
        assert isinstance(losses["bbox_regression"], Tensor)
        assert isinstance(losses["objectness"], Tensor)


@pytest.mark.parametrize("arch", ["yolov5s", "yolov5m", "yolov5l", "yolotr"])
def test_torchscript(arch):
    model = models.__dict__[arch](pretrained=True, size=(320, 320), score_thresh=0.45)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.eval()

    x = [torch.rand(3, 288, 320), torch.rand(3, 300, 256)]

    out = model(x)
    out_script = scripted_model(x)

    torch.testing.assert_close(
        out[0]["scores"], out_script[1][0]["scores"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out[0]["labels"], out_script[1][0]["labels"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out[0]["boxes"], out_script[1][0]["boxes"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "arch, version, hash_prefix", [("yolov5s", "v4.0", "9ca9a642")]
)
def test_load_from_yolov5(arch, version, hash_prefix):
    img_path = "test/assets/bus.jpg"
    yolov5s_r40_path = Path(f"{arch}.pt")

    if not yolov5s_r40_path.exists():
        torch.hub.download_url_to_file(
            f"https://github.com/ultralytics/yolov5/releases/download/{version}/{arch}.pt",
            yolov5s_r40_path,
            hash_prefix=hash_prefix,
        )

    yolov5 = YOLOv5()
    model_yolov5 = yolov5.load_from_yolov5(yolov5s_r40_path, score_thresh=0.25)
    model_yolov5.eval()
    out_from_yolov5 = model_yolov5.predict(img_path)
    assert isinstance(out_from_yolov5[0], dict)
    assert isinstance(out_from_yolov5[0]["boxes"], Tensor)
    assert isinstance(out_from_yolov5[0]["labels"], Tensor)
    assert isinstance(out_from_yolov5[0]["scores"], Tensor)

    model = models.__dict__[arch](pretrained=True, score_thresh=0.25)
    model.eval()
    out = model.predict(img_path)

    torch.testing.assert_close(
        out_from_yolov5[0]["scores"], out[0]["scores"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_from_yolov5[0]["labels"], out[0]["labels"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_from_yolov5[0]["boxes"], out[0]["boxes"], rtol=0, atol=0
    )
