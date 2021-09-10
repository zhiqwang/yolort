import torch
from yolort.models.anchor_utils import AnchorGenerator


class TestAnchorGenerator:
    strides = [4]
    anchor_grids = [[6, 14]]

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [torch.rand(2, 8, s0 // 5, s1 // 5)]
        return features

    def test_anchor_generator(self):
        images = torch.randn(2, 3, 10, 10)
        features = self.get_features(images)
        model = AnchorGenerator(self.strides, self.anchor_grids)
        model.eval()
        anchors = model(features)

        expected_anchor_output = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        expected_wh_output = torch.tensor([[4.], [4.], [4.], [4.]])
        expected_xy_output = torch.tensor([[6., 14.], [6., 14.], [6., 14.], [6., 14.]])

        assert len(anchors) == 3
        assert tuple(anchors[0].shape) == (4, 2)
        assert tuple(anchors[1].shape) == (4, 1)
        assert tuple(anchors[2].shape) == (4, 2)

        torch.testing.assert_close(anchors[0], expected_anchor_output, rtol=0, atol=0)
        torch.testing.assert_close(anchors[1], expected_wh_output, rtol=0, atol=0)
        torch.testing.assert_close(anchors[2], expected_xy_output, rtol=0, atol=0)
