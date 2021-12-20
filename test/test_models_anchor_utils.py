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
        images = torch.rand(2, 3, 10, 10)
        features = self.get_features(images)
        model = AnchorGenerator(self.strides, self.anchor_grids)
        model.eval()
        anchors = model(features)

        expected_grids = torch.tensor([[[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]]])
        expected_shifts = torch.tensor([[[[[6.0, 14.0], [6.0, 14.0]], [[6.0, 14.0], [6.0, 14.0]]]]])

        assert len(anchors) == 2
        assert len(anchors[0]) == len(anchors[1]) == 1
        assert tuple(anchors[0][0].shape) == (1, 1, 2, 2, 2)
        assert tuple(anchors[1][0].shape) == (1, 1, 2, 2, 2)

        torch.testing.assert_close(anchors[0][0], expected_grids)
        torch.testing.assert_close(anchors[1][0], expected_shifts)
