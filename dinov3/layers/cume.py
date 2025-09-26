# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
from typing import Tuple
from torch.nn import functional as F
import math

def localized_matching(x, r, use_euc_dist=False):
    t = x.shape[1]
    r = min(r, t // 2)

    with torch.no_grad():
        B,N,_ = x.shape
        if use_euc_dist:
            scores = -torch.linalg.vector_norm(x[..., :-1, :] - x[..., 1:, :], dim=-1)
        else:
            x = x / x.norm(dim=-1, keepdim=True)
            scores = torch.einsum('bnc,bnc->bn', x[..., :-1, :], x[..., 1:, :])
        if scores.shape[-1]%2:
            scores = F.pad(scores, (0, 1), value=-math.inf)
        scores = scores.reshape(B, -1, 2)

        node_max, node_idx = scores.max(dim=-1)
        src_idx = node_max.argsort(dim=-1, descending=True)[..., :r, None]
        dst_idx = (node_idx[..., None].gather(dim=-2, index=src_idx) + src_idx) * 2
        src_idx = src_idx*2 + 1

        mask = torch.ones((B,N,1), dtype=torch.bool, device=src_idx.device)
        mask.scatter_(1, src_idx, 0)
            
    def merge(x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        src = x.gather(dim=-2, index=src_idx.expand(-1, -1, c))
        dst = x.gather(dim=-2, index=dst_idx.expand(-1, -1, c))
        # scatter reduce does not support max magnitude, so we have to break into 2 steps
        src[..., ::2, :] = torch.where(torch.abs(src[..., ::2, :])>torch.abs(dst[..., ::2, :]), src[..., ::2, :], dst[..., ::2, :])
        x.scatter_(dim=-2, index=dst_idx[..., ::2, :].expand(-1, -1, c), src=src[..., ::2, :])
        src[..., 1::2, :] = torch.where(torch.abs(src[..., 1::2, :])>torch.abs(dst[..., 1::2, :]), src[..., 1::2, :], dst[..., 1::2, :])
        x.scatter_(dim=-2, index=dst_idx[..., 1::2, :].expand(-1, -1, c), src=src[..., 1::2, :])
        x = x[mask.expand(-1, -1, c)].reshape(b, -1, c)
        return x

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.masked_scatter_(mask, x.reshape(-1))
        src = out.gather(dim=-2, index=dst_idx.expand(-1, -1, c))
        out = torch.scatter(out, dim=-2, index=src_idx.expand(-1, -1, c), src=src)
        return out

    return merge, unmerge

class CubistMerger:
    def __init__(self):
        self.merge_func = []
        self.unmerge_func = []

    def init(self, hw):
        self.h, self.w = hw
        self.merge_func = []
        self.unmerge_func = []

    def semi_structured_merge(self, x, r, vertical=False):
        if r <= 0:
            return x
        if not vertical:
            x = x.reshape(self.b*self.h, self.w, self.C)
            merge, unmerge = localized_matching(x, r)
            x = merge(x)
            self.w = self.w-r
            x = x.reshape(self.b, self.h, self.w, self.C)
        else:
            x = x.transpose(1, 2).reshape(self.b*self.w, self.h, self.C)
            merge, unmerge = localized_matching(x, r)
            x = merge(x)
            self.h = self.h-r
            x = x.reshape(self.b, self.w, self.h, self.C).transpose(1, 2)

        self.merge_func.append((vertical, merge))
        self.unmerge_func.append((vertical, unmerge))
        return x

    def semi_structured_unmerge(self, x, unmerge, vertical=False):
        if not vertical:
            x = x.reshape(self.b*self.h, self.w, self.C)
            x = unmerge(x)
            self.w = x.shape[1]
            x = x.reshape(self.b, self.h, self.w, self.C)
        else:
            x = x.transpose(1, 2).reshape(self.b*self.w, self.h, self.C)
            x = unmerge(x)
            self.h = x.shape[1]
            x = x.reshape(self.b, self.w, self.h, self.C).transpose(1, 2)
        return x

    def semi_structured_merge_2d(self, x, r_h, r_w):
        x = self.semi_structured_merge(x, r_w)
        x = self.semi_structured_merge(x, r_h, vertical=True)
        return x

    def semi_structured_unmerge_2d(self, x):
        for (vertical, unmerge) in self.unmerge_func[::-1]:
            x = self.semi_structured_unmerge(x, unmerge, vertical=vertical)
        return x

    def merge(self, x, r_h, r_w):
        self.b = x.shape[0]
        self.C = x.shape[-1]
        x = x.reshape(self.b, self.h, self.w, self.C)
        x = self.semi_structured_merge_2d(x, r_h, r_w)
        x = x.reshape(self.b, -1, self.C)
        return x

    def recover(self, x):
        h, w = self.h, self.w
        x = x.reshape(self.b, self.h, self.w, self.C)
        x = self.semi_structured_unmerge_2d(x)
        x = x.reshape(self.b, -1, self.C)
        self.h, self.w = h, w
        return x