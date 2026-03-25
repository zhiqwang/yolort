# Copyright (c) 2021, yolort team. All rights reserved.
import pytest
import torch

from yolort.v5.models.common import (
    Bottleneck,
    C3,
    Concat,
    Contract,
    Conv,
    DWConv,
    Expand,
    Flatten,
    Focus,
    GhostBottleneck,
    GhostConv,
    SPP,
    SPPF,
    autopad,
)


class TestAutopad:
    def test_int_kernel(self):
        assert autopad(3) == 1
        assert autopad(5) == 2
        assert autopad(1) == 0

    def test_int_kernel_even(self):
        assert autopad(4) == 2

    def test_explicit_padding(self):
        assert autopad(3, p=0) == 0
        assert autopad(5, p=1) == 1

    def test_list_kernel(self):
        result = autopad([3, 5])
        assert result == [1, 2]


class TestConv:
    def test_output_shape(self):
        conv = Conv(3, 16, k=3, s=1)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)

    def test_stride_2(self):
        conv = Conv(3, 16, k=3, s=2)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 16, 16)

    def test_version_r31(self):
        conv = Conv(3, 16, k=3, s=1, version="r3.1")
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)

    def test_invalid_version(self):
        with pytest.raises(NotImplementedError):
            Conv(3, 16, k=3, s=1, version="r99.0")

    def test_no_activation(self):
        conv = Conv(3, 16, k=1, act=False)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)

    def test_forward_fuse(self):
        conv = Conv(3, 16, k=3, s=1)
        x = torch.randn(1, 3, 32, 32)
        out = conv.forward_fuse(x)
        assert out.shape == (1, 16, 32, 32)


class TestDWConv:
    def test_output_shape(self):
        dwconv = DWConv(16, 16, k=3, s=1)
        x = torch.randn(1, 16, 32, 32)
        out = dwconv(x)
        assert out.shape == (1, 16, 32, 32)


class TestBottleneck:
    def test_with_shortcut(self):
        bn = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 16, 16)
        out = bn(x)
        assert out.shape == (1, 64, 16, 16)

    def test_without_shortcut(self):
        bn = Bottleneck(64, 32, shortcut=False)
        x = torch.randn(1, 64, 16, 16)
        out = bn(x)
        assert out.shape == (1, 32, 16, 16)


class TestC3:
    def test_output_shape(self):
        c3 = C3(64, 64, n=1)
        x = torch.randn(1, 64, 16, 16)
        out = c3(x)
        assert out.shape == (1, 64, 16, 16)

    def test_different_channels(self):
        c3 = C3(32, 64, n=2)
        x = torch.randn(1, 32, 16, 16)
        out = c3(x)
        assert out.shape == (1, 64, 16, 16)


class TestSPP:
    def test_output_shape(self):
        spp = SPP(64, 64)
        x = torch.randn(1, 64, 16, 16)
        out = spp(x)
        assert out.shape == (1, 64, 16, 16)


class TestSPPF:
    def test_output_shape(self):
        sppf = SPPF(64, 64)
        x = torch.randn(1, 64, 16, 16)
        out = sppf(x)
        assert out.shape == (1, 64, 16, 16)


class TestFocus:
    def test_output_shape(self):
        focus = Focus(3, 64, k=3)
        x = torch.randn(1, 3, 64, 64)
        out = focus(x)
        assert out.shape == (1, 64, 32, 32)


class TestConcat:
    def test_concat_list(self):
        concat = Concat(dimension=1)
        x = [torch.randn(1, 16, 8, 8), torch.randn(1, 32, 8, 8)]
        out = concat(x)
        assert out.shape == (1, 48, 8, 8)

    def test_concat_single_tensor(self):
        concat = Concat(dimension=1)
        x = torch.randn(1, 16, 8, 8)
        out = concat(x)
        assert out.shape == (1, 16, 8, 8)

    def test_concat_dim0(self):
        concat = Concat(dimension=0)
        x = [torch.randn(1, 16, 8, 8), torch.randn(2, 16, 8, 8)]
        out = concat(x)
        assert out.shape == (3, 16, 8, 8)


class TestFlatten:
    def test_flatten(self):
        flatten = Flatten()
        x = torch.randn(4, 256, 1, 1)
        out = flatten(x)
        assert out.shape == (4, 256)


class TestContract:
    def test_output_shape(self):
        contract = Contract(gain=2)
        x = torch.randn(1, 64, 80, 80)
        out = contract(x)
        assert out.shape == (1, 256, 40, 40)

    def test_gain_4(self):
        contract = Contract(gain=4)
        x = torch.randn(1, 16, 80, 80)
        out = contract(x)
        assert out.shape == (1, 256, 20, 20)


class TestExpand:
    def test_output_shape(self):
        expand = Expand(gain=2)
        x = torch.randn(1, 64, 80, 80)
        out = expand(x)
        assert out.shape == (1, 16, 160, 160)

    def test_contract_expand_roundtrip(self):
        contract = Contract(gain=2)
        expand = Expand(gain=2)
        x = torch.randn(1, 64, 80, 80)
        y = expand(contract(x))
        assert y.shape == x.shape


class TestGhostConv:
    def test_output_shape(self):
        gc = GhostConv(32, 64)
        x = torch.randn(1, 32, 16, 16)
        out = gc(x)
        assert out.shape == (1, 64, 16, 16)


class TestGhostBottleneck:
    def test_output_shape(self):
        gb = GhostBottleneck(64, 64)
        x = torch.randn(1, 64, 16, 16)
        out = gb(x)
        assert out.shape == (1, 64, 16, 16)
