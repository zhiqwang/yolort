# Copyright (c) 2021, yolort team. All rights reserved.
import torch

from yolort.v5.utils.activations import Hardswish, SiLU


class TestSiLU:
    def test_output_shape(self):
        silu = SiLU()
        x = torch.randn(2, 3, 4, 4)
        out = silu(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        silu = SiLU()
        x = torch.zeros(1, 3, 4, 4)
        out = silu(x)
        assert torch.all(out == 0)

    def test_positive_input(self):
        silu = SiLU()
        x = torch.tensor([1.0])
        out = silu(x)
        expected = 1.0 * torch.sigmoid(torch.tensor([1.0]))
        torch.testing.assert_close(out, expected)

    def test_negative_input(self):
        silu = SiLU()
        x = torch.tensor([-1.0])
        out = silu(x)
        expected = -1.0 * torch.sigmoid(torch.tensor([-1.0]))
        torch.testing.assert_close(out, expected)


class TestHardswish:
    def test_output_shape(self):
        hs = Hardswish()
        x = torch.randn(2, 3, 4, 4)
        out = hs(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        hs = Hardswish()
        x = torch.zeros(1, 3, 4, 4)
        out = hs(x)
        # hardswish(0) = 0 * hardtanh(3)/6 = 0 * 3/6 = 0
        assert torch.all(out == 0)

    def test_large_positive(self):
        hs = Hardswish()
        x = torch.tensor([10.0])
        out = hs(x)
        # For large positive x: hardtanh(x+3, 0, 6) = 6, so result = x * 6/6 = x
        torch.testing.assert_close(out, x)

    def test_large_negative(self):
        hs = Hardswish()
        x = torch.tensor([-10.0])
        out = hs(x)
        # For large negative x: hardtanh(x+3, 0, 6) = 0, so result = 0
        assert out.item() == 0.0
