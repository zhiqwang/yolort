# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
import torch
from torch import nn, Tensor
from typing import Tuple, List


class AnchorGenerator(nn.Module):

    def __init__(
        self,
        strides: List[int],
        anchor_grids: List[List[int]],
    ):
        super().__init__()
        assert len(strides) == len(anchor_grids)
        self.num_anchors = len(anchor_grids)
        self.strides = strides
        self.anchor_grids = anchor_grids

    def set_wh_weights(
        self,
        grid_sizes: List[List[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:

        wh_weights = []

        for size, stride in zip(grid_sizes, self.strides):
            grid_height, grid_width = size
            stride = torch.as_tensor([stride], dtype=dtype, device=device)
            stride = stride.view(-1, 1)
            stride = stride.repeat(1, grid_height * grid_width * self.num_anchors)
            stride = stride.reshape(-1, 1)
            wh_weights.append(stride)

        return torch.cat(wh_weights)

    def set_xy_weights(
        self,
        grid_sizes: List[List[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:

        xy_weights = []

        for size, anchor_grid in zip(grid_sizes, self.anchor_grids):
            grid_height, grid_width = size
            anchor_grid = torch.as_tensor(anchor_grid, dtype=dtype, device=device)
            anchor_grid = anchor_grid.view(-1, 2)
            anchor_grid = anchor_grid.repeat(1, grid_height * grid_width)
            anchor_grid = anchor_grid.reshape(-1, 2)
            xy_weights.append(anchor_grid)

        return torch.cat(xy_weights)

    def grid_anchors(
        self,
        grid_sizes: List[List[int]],
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:

        anchors = []

        for size in grid_sizes:
            grid_height, grid_width = size

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

            shifts = torch.stack((shift_x, shift_y), dim=2)
            shifts = shifts.view(1, grid_height, grid_width, 2)
            shifts = shifts.repeat(self.num_anchors, 1, 1, 1)
            shifts = shifts - torch.tensor(0.5, dtype=shifts.dtype, device=device)
            shifts = shifts.reshape(-1, 2)

            anchors.append(shifts)

        return torch.cat(anchors)

    def forward(self, feature_maps: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        wh_weights = self.set_wh_weights(grid_sizes, dtype, device)
        xy_weights = self.set_xy_weights(grid_sizes, dtype, device)
        anchors = self.grid_anchors(grid_sizes, device)

        return anchors, wh_weights, xy_weights
