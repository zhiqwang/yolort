import contextlib
import io
import os
import warnings

import pytest
import torch
from torch import Tensor
from yolort import models
from yolort.models import YOLO, YOLOv5
from yolort.models.anchor_utils import AnchorGenerator
from yolort.models.backbone_utils import darknet_pan_backbone
from yolort.models.box_head import YOLOHead, PostProcess, SetCriterion
from yolort.models.transformer import darknet_tan_backbone
from yolort.models.yolo_lite import yolov5_mobilenet_v3_small_fpn
from yolort.v5 import get_yolov5_size, attempt_download


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
            torch.testing.assert_close(results, results_from_imported, atol=tol, rtol=tol)
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

    num_classes = 80
    num_outputs = num_classes + 5

    @staticmethod
    def _get_in_channels(width_multiple, use_p6):
        grow_widths = [256, 512, 768, 1024] if use_p6 else [256, 512, 1024]
        in_channels = [int(gw * width_multiple) for gw in grow_widths]
        return in_channels

    @staticmethod
    def _get_strides(use_p6: bool):
        if use_p6:
            strides = [8, 16, 32, 64]
        else:
            strides = [8, 16, 32]
        return strides

    @staticmethod
    def _get_anchor_grids(use_p6: bool):
        if use_p6:
            anchor_grids = [
                [19, 27, 44, 40, 38, 94],
                [96, 68, 86, 152, 180, 137],
                [140, 301, 303, 264, 238, 542],
                [436, 615, 739, 380, 925, 792],
            ]
        else:
            anchor_grids = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        return anchor_grids

    def _compute_anchors(self, height, width, use_p6: bool):
        strides = self._get_strides(use_p6)
        anchors_num = len(strides)
        anchors_shape = []
        for s in strides:
            anchors_shape.append((height // s, width // s))
        return anchors_num, anchors_shape

    def _get_feature_shapes(self, height, width, width_multiple=0.5, use_p6=False):
        in_channels = self._get_in_channels(width_multiple, use_p6)
        strides = self._get_strides(use_p6)

        return [(c, height // s, width // s) for (c, s) in zip(in_channels, strides)]

    def _get_feature_maps(self, batch_size, height, width, width_multiple=0.5, use_p6=False):
        feature_shapes = self._get_feature_shapes(
            height,
            width,
            width_multiple=width_multiple,
            use_p6=use_p6,
        )
        feature_maps = [torch.rand(batch_size, *f_shape) for f_shape in feature_shapes]
        return feature_maps

    def _get_head_outputs(self, batch_size, height, width, width_multiple=0.5, use_p6=False):
        feature_shapes = self._get_feature_shapes(
            height,
            width,
            width_multiple=width_multiple,
            use_p6=use_p6,
        )

        num_outputs = self.num_outputs
        head_shapes = [(batch_size, 3, *f_shape[1:], num_outputs) for f_shape in feature_shapes]
        head_outputs = [torch.rand(*h_shape) for h_shape in head_shapes]

        return head_outputs

    def _init_test_backbone_with_pan(
        self,
        depth_multiple,
        width_multiple,
        version,
        use_p6,
        use_tan,
    ):
        model_size = get_yolov5_size(depth_multiple, width_multiple)
        backbone_name = f"darknet_{model_size}_{version.replace('.', '_')}"
        backbone_arch = eval(f"darknet_{'tan' if use_tan else 'pan'}_backbone")
        assert backbone_arch in [darknet_pan_backbone, darknet_tan_backbone]
        model = backbone_arch(
            backbone_name,
            depth_multiple,
            width_multiple,
            version=version,
            use_p6=use_p6,
        )
        return model

    @pytest.mark.parametrize(
        "depth_multiple, width_multiple, version, use_p6, use_tan",
        [
            (0.33, 0.5, "r4.0", False, True),
            (0.33, 0.5, "r3.1", False, False),
            (0.33, 0.5, "r4.0", False, False),
            (0.33, 0.5, "r6.0", False, False),
            (0.33, 0.5, "r6.0", True, False),
            (0.67, 0.75, "r6.0", False, False),
        ],
    )
    @pytest.mark.parametrize("batch_size, height, width", [(4, 448, 320), (2, 384, 640)])
    def test_backbone_with_pan(
        self,
        depth_multiple,
        width_multiple,
        version,
        use_p6,
        use_tan,
        batch_size,
        height,
        width,
    ):
        out_shape = self._get_feature_shapes(height, width, width_multiple=width_multiple, use_p6=use_p6)

        x = torch.rand(batch_size, 3, height, width)
        model = self._init_test_backbone_with_pan(
            depth_multiple, width_multiple, version, use_p6, use_tan=use_tan
        )
        out = model(x)

        expected_num_output = 4 if use_p6 else 3
        assert len(out) == expected_num_output
        for i in range(expected_num_output):
            assert tuple(out[i].shape) == (batch_size, *out_shape[i])

        _check_jit_scriptable(model, (x,))

    def _init_test_anchor_generator(self, use_p6=False):
        strides = self._get_strides(use_p6)
        anchor_grids = self._get_anchor_grids(use_p6)
        anchor_generator = AnchorGenerator(strides, anchor_grids)
        return anchor_generator

    @pytest.mark.parametrize(
        "width_multiple, use_p6",
        [(0.5, False), (0.5, True)],
    )
    @pytest.mark.parametrize("batch_size, height, width", [(4, 448, 320), (2, 384, 640)])
    def test_anchor_generator(self, width_multiple, use_p6, batch_size, height, width):
        feature_maps = self._get_feature_maps(
            batch_size, height, width, width_multiple=width_multiple, use_p6=use_p6
        )
        model = self._init_test_anchor_generator(use_p6)
        anchors = model(feature_maps)
        expected_anchors_num, expected_anchors_shape = self._compute_anchors(height, width, use_p6)

        assert len(anchors) == 2
        assert len(anchors[0]) == len(anchors[1]) == expected_anchors_num
        for i in range(expected_anchors_num):
            assert tuple(anchors[0][i].shape) == (1, 3, *(expected_anchors_shape[i]), 2)
            assert tuple(anchors[1][i].shape) == (1, 3, *(expected_anchors_shape[i]), 2)

        _check_jit_scriptable(model, (feature_maps,))

    def _init_test_yolo_head(self, width_multiple=0.5, use_p6=False):
        in_channels = self._get_in_channels(width_multiple, use_p6)
        strides = self._get_strides(use_p6)
        num_anchors = len(strides)
        num_classes = self.num_classes

        box_head = YOLOHead(in_channels, num_anchors, strides, num_classes)
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

    def _init_test_postprocessors(self, strides):
        score_thresh = 0.5
        nms_thresh = 0.45
        detections_per_img = 100
        postprocessors = PostProcess(strides, score_thresh, nms_thresh, detections_per_img)
        return postprocessors

    @pytest.mark.parametrize("use_p6", [False, True])
    def test_postprocessors(self, use_p6):
        N, H, W = 4, 416, 352
        strides = self._get_strides(use_p6)
        feature_maps = self._get_feature_maps(N, H, W, use_p6=use_p6)
        head_outputs = self._get_head_outputs(N, H, W, use_p6=use_p6)

        anchor_generator = self._init_test_anchor_generator(use_p6=use_p6)
        grids, shifts = anchor_generator(feature_maps)
        model = self._init_test_postprocessors(strides)
        out = model(head_outputs, grids, shifts)

        assert len(out) == N
        assert isinstance(out[0], dict)
        assert isinstance(out[0]["boxes"], Tensor)
        assert isinstance(out[0]["labels"], Tensor)
        assert isinstance(out[0]["scores"], Tensor)
        _check_jit_scriptable(model, (head_outputs, grids, shifts))

    def test_criterion(self, use_p6=False):
        N, H, W = 4, 640, 640
        head_outputs = self._get_head_outputs(N, H, W)
        strides = self._get_strides(use_p6)
        anchor_grids = self._get_anchor_grids(use_p6)
        num_classes = self.num_classes

        targets = torch.tensor(
            [
                [0.0000, 7.0000, 0.0714, 0.3749, 0.0760, 0.0654],
                [0.0000, 1.0000, 0.1027, 0.4402, 0.2053, 0.1920],
                [1.0000, 5.0000, 0.4720, 0.6720, 0.3280, 0.1760],
                [3.0000, 3.0000, 0.6305, 0.3290, 0.3274, 0.2270],
            ]
        )
        criterion = SetCriterion(strides, anchor_grids, num_classes)
        losses = criterion(targets, head_outputs)
        assert isinstance(losses, dict)
        assert isinstance(losses["cls_logits"], Tensor)
        assert isinstance(losses["bbox_regression"], Tensor)
        assert isinstance(losses["objectness"], Tensor)


@pytest.mark.parametrize("arch", ["yolov5s", "yolov5m", "yolov5l", "yolov5ts"])
def test_torchscript(arch):
    model = models.__dict__[arch](pretrained=True, size=(320, 320), score_thresh=0.45)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.eval()

    x = [torch.rand(3, 288, 320), torch.rand(3, 300, 256)]

    out = model(x)
    out_script = scripted_model(x)

    torch.testing.assert_close(out[0]["scores"], out_script[1][0]["scores"], rtol=0, atol=0)
    torch.testing.assert_close(out[0]["labels"], out_script[1][0]["labels"], rtol=0, atol=0)
    torch.testing.assert_close(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "arch, size_divisible, version, upstream_version, hash_prefix",
    [
        ("yolov5s", 32, "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", 32, "r6.0", "v6.0", "649e089f"),
        ("yolov5s", 32, "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", 64, "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_load_from_yolov5(arch, size_divisible, version, upstream_version, hash_prefix):
    img_path = "test/assets/bus.jpg"

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    score_thresh = 0.25

    model_yolov5 = YOLOv5.load_from_yolov5(
        checkpoint_path,
        score_thresh=score_thresh,
        version=version,
        size_divisible=size_divisible,
    )
    model_yolov5.eval()
    out_from_yolov5 = model_yolov5.predict(img_path)
    assert isinstance(out_from_yolov5[0], dict)
    assert isinstance(out_from_yolov5[0]["boxes"], Tensor)
    assert isinstance(out_from_yolov5[0]["labels"], Tensor)
    assert isinstance(out_from_yolov5[0]["scores"], Tensor)

    model = models.__dict__[arch](
        upstream_version=version,
        pretrained=True,
        score_thresh=score_thresh,
    )
    model.eval()
    out = model.predict(img_path)

    torch.testing.assert_close(out_from_yolov5[0]["scores"], out[0]["scores"], rtol=0, atol=0)
    torch.testing.assert_close(out_from_yolov5[0]["labels"], out[0]["labels"], rtol=0, atol=0)
    torch.testing.assert_close(out_from_yolov5[0]["boxes"], out[0]["boxes"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "arch, size_divisible, version, upstream_version, hash_prefix",
    [
        ("yolov5s", 32, "r4.0", "v4.0", "9ca9a642"),
        ("yolov5n", 32, "r6.0", "v6.0", "649e089f"),
        ("yolov5s", 32, "r6.0", "v6.0", "c3b140f3"),
        ("yolov5n6", 64, "r6.0", "v6.0", "beecbbae"),
    ],
)
def test_load_from_yolov5_torchscript(arch, size_divisible, version, upstream_version, hash_prefix):
    import cv2
    from yolort.utils import read_image_to_tensor
    from yolort.v5 import letterbox

    # Loading and pre-processing the image
    img_path = "test/assets/zidane.jpg"
    img_raw = cv2.imread(img_path)
    img = letterbox(img_raw, new_shape=(640, 640), stride=size_divisible)[0]
    img = read_image_to_tensor(img)

    base_url = "https://github.com/ultralytics/yolov5/releases/download/"
    model_url = f"{base_url}/{upstream_version}/{arch}.pt"
    checkpoint_path = attempt_download(model_url, hash_prefix=hash_prefix)

    score_thresh = 0.25

    model = YOLO.load_from_yolov5(checkpoint_path, score_thresh=score_thresh, version=version)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.eval()

    out = model(img[None])
    out_script = scripted_model(img[None])

    torch.testing.assert_close(out[0]["scores"], out_script[1][0]["scores"], rtol=0, atol=0)
    torch.testing.assert_close(out[0]["labels"], out_script[1][0]["labels"], rtol=0, atol=0)
    torch.testing.assert_close(out[0]["boxes"], out_script[1][0]["boxes"], rtol=0, atol=0)


def test_yolov5_mobilenet_v3_small_fpn():

    model = yolov5_mobilenet_v3_small_fpn()
    model = model.eval()

    images = torch.rand(4, 3, 320, 320)
    out = model(images)
    assert isinstance(out, list)
    assert len(out) == 4
    assert isinstance(out[0], dict)
    assert isinstance(out[0]["boxes"], Tensor)
    assert isinstance(out[0]["labels"], Tensor)
    assert isinstance(out[0]["scores"], Tensor)
