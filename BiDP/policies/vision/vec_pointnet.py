import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
import logging

from policies.vision.vec_layers import VecLinear
from policies.vision.vec_layers import VecLinNormAct as VecLNA


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VecPointNet(nn.Module):
    def __init__(
        self,
        h_dim=128,
        c_dim=128,
        num_layers=4,
        knn=16,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.c_dim = c_dim
        self.num_layers = num_layers
        self.knn = knn

        self.pool = meanpool

        act_func = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        vnla_cfg = {"mode": "so3", "act_func": act_func}

        self.conv_in = VecLNA(3, h_dim, **vnla_cfg)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(VecLNA(h_dim, h_dim, **vnla_cfg))
            self.global_layers.append(VecLNA(h_dim * 2, h_dim, **vnla_cfg))
        self.conv_out = VecLinear(h_dim * self.num_layers, c_dim, mode="so3")

        self.fc_inv = VecLinear(c_dim, 3, mode="so3")

    def get_graph_feature(self, x: torch.Tensor, k: int, knn_idx=None, cross=False):
        # x: B,C,3,N return B,C*2,3,N,K


        B, C, _, N = x.shape
        if knn_idx is None:
            # if knn_idx is not none, compute the knn by x distance; ndf use fixed knn as input topo
            _x = x.reshape(B, -1, N)
            _, knn_idx, neighbors = knn_points(
                _x.transpose(2, 1), _x.transpose(2, 1), K=k, return_nn=True
            )  # B,N,K; B,N,K; B,N,K,D
            neighbors = neighbors.reshape(B, N, k, C, 3).permute(0, -2, -1, 1, 2)
        else:  # gather from the input knn idx
            assert knn_idx.shape[-1] == k, f"input knn gather idx should have k={k}"
            neighbors = torch.gather(
                x[..., None, :].expand(-1, -1, -1, N, -1),
                dim=-1,
                index=knn_idx[:, None, None, ...].expand(-1, C, 3, -1, -1),
            )  # B,C,3,N,K
        x_padded = x[..., None].expand_as(neighbors)

        if cross:
            x_dir = F.normalize(x, dim=2)
            x_dir_padded = x_dir[..., None].expand_as(neighbors)
            cross = torch.cross(x_dir_padded, neighbors, dim=2)
            y = torch.cat([cross, neighbors - x_padded, x_padded], 1)
        else:
            y = torch.cat([neighbors - x_padded, x_padded], 1)
        return y, knn_idx  # B,C*2,3,N,K

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 3, N]

        x, knn_idx = self.get_graph_feature(x, self.knn, cross=True)
        x, _ = self.conv_in(x)
        x = self.pool(x)

        y = x
        feat_list = []
        for i in range(self.num_layers):
            y, _ = self.layers[i](y)
            y_global = y.mean(-1, keepdim=True)
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y, _ = self.global_layers[i](y)
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x, _ = self.conv_out(x)

        return x.mean(-1), x
