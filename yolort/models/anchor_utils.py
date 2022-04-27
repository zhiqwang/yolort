# Copyright (c) 2020, yolort team. All rights reserved.

from typing import Tuple, List

import torch
from torch import nn, Tensor


class AnchorGenerator(nn.Module):
    def __init__(self, strides: List[int], anchor_grids: List[List[float]]):

        super().__init__()
        assert len(strides) == len(anchor_grids)
        self.strides = strides
        self.anchor_grids = anchor_grids
        self.num_layers = len(anchor_grids)
        self.num_anchors = len(anchor_grids[0]) // 2

    def _generate_grids(
        self,
        grid_sizes: List[List[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:

        grids = []
        for height, width in grid_sizes:
            # For output anchor, compute [x_center, y_center, x_center, y_center]
            widths = torch.arange(width, dtype=torch.int32, device=device).to(dtype=dtype)
            heights = torch.arange(height, dtype=torch.int32, device=device).to(dtype=dtype)

            shift_y, shift_x = torch.meshgrid(heights, widths)

            grid = torch.stack((shift_x, shift_y), 2).expand((1, self.num_anchors, height, width, 2))
            grids.append(grid)

        return grids

    def _generate_shifts(
        self,
        grid_sizes: List[List[int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:

        anchors = torch.as_tensor(self.anchor_grids, dtype=torch.float32, device=device).to(dtype=dtype)
        strides = torch.as_tensor(self.strides, dtype=torch.float32, device=device).to(dtype=dtype)
        anchors = anchors.view(self.num_layers, -1, 2) / strides.view(-1, 1, 1)

        shifts = []
        for i, (height, width) in enumerate(grid_sizes):
            shift = (
                (anchors[i].clone() * self.strides[i])
                .view((1, self.num_anchors, 1, 1, 2))
                .expand((1, self.num_anchors, height, width, 2))
                .contiguous()
                .to(dtype=dtype)
            )
            shifts.append(shift)
        return shifts

    def forward(self, feature_maps: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        grids = self._generate_grids(grid_sizes, dtype=dtype, device=device)
        shifts = self._generate_shifts(grid_sizes, dtype=dtype, device=device)
        return grids, shifts
