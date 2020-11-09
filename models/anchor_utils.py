# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)
import torch
from torch import nn, Tensor
from torch.jit.annotations import Tuple, List, Dict, Optional


class AnchorGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, num_anchors, strides, anchor_grids):
        super().__init__()

        if not isinstance(strides[0], (list, tuple)):
            strides = tuple((s,) for s in strides)

        self.num_anchors = num_anchors
        self.strides = strides
        self.anchor_grids = anchor_grids
        self._cache = {}

    def set_wh_weights(self, grid_sizes, dtype, device):
        # type: (int, Device) -> Tensor  # noqa: F821
        wh_weights_per_image = []
        for grid_size, stride in zip(grid_sizes, self.strides):
            wh_weights_per_image.extend([stride] * (grid_size[0] * grid_size[1] * self.num_anchors))

        wh_weights_per_image = torch.as_tensor(wh_weights_per_image, dtype=dtype, device=device)
        wh_weights_per_image = wh_weights_per_image.reshape(-1, 1)

        return wh_weights_per_image

    def set_xy_weights(self, grid_sizes, dtype, device):
        # type: (int, Device) -> Tensor  # noqa: F821
        xy_weights_per_image = []
        for grid_size, anchor_grid in zip(grid_sizes, self.anchor_grids):
            anchor_grid = torch.as_tensor(anchor_grid, dtype=dtype, device=device)
            anchor_grid = anchor_grid.view(-1, 1, 1, 2)
            anchor_grid = anchor_grid.repeat(1, grid_size[0], grid_size[1], 1)
            anchor_grid = anchor_grid.reshape(-1, 2)
            xy_weights_per_image.append(anchor_grid)

        return torch.cat(xy_weights_per_image)

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, device):
        # type: (List[List[int]], Device) -> List[Tensor]  # noqa: F821
        anchors = []

        for size in grid_sizes:
            grid_height, grid_width = size

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

            shifts = torch.stack((shift_x, shift_y), dim=2)
            shifts = shifts.view(1, grid_height, grid_width, 2)
            shifts = shifts.repeat(3, 1, 1, 1)
            shifts = shifts - 0.5

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(shifts.reshape(-1, 2))

        return anchors

    def cached_grid_anchors(self, grid_sizes, device):
        # type: (List[List[int]], Device) -> List[Tensor]  # noqa: F821
        key = str(grid_sizes)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, device)
        self._cache[key] = anchors
        return anchors

    def forward(self, feature_maps):
        # type: (List[Tensor]) -> Tuple[Tensor, List[Tensor]]
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        wh_weights = self.set_wh_weights(grid_sizes, dtype, device)
        xy_weights = self.set_xy_weights(grid_sizes, dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, device)
        anchors_in_image = torch.jit.annotate(List[torch.Tensor], [])
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors_in_image.append(anchors_per_feature_map)
        anchors = torch.cat(anchors_in_image)

        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors, wh_weights, xy_weights
