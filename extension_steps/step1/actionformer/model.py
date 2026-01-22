from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal positional encoding (Transformer-style).
# Named "sinusoid" because each position index is mapped to a vector built from
# sine and cosine waves at different frequencies:
#   - even dims: sin(pos / 10000^(2i/d))
#   - odd  dims: cos(pos / 10000^(2i/d))
# This injects timestep/order information into the model (self-attention alone
# does not know sequence order). The name comes from the original Transformer
# positional encoding popularized in "Attention Is All You Need".
def get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    def get_position_angle_vec(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class LayerNorm(nn.Module):
    """LayerNorm over channels for (B, C, T)."""

    def __init__(self, n_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, n_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.gamma + self.beta


class MaskedConv1D(nn.Module):
    """1D convolution with mask propagation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T), mask: (B, 1, T) boolean
        x = x * mask.to(x.dtype)
        out = self.conv(x)
        out_mask = F.interpolate(mask.float(), size=out.size(-1), mode="nearest").bool()
        return out * out_mask.to(out.dtype), out_mask


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_head: int,
        attn_pdrop: float = 0.0,
        proj_pdrop: float = 0.0,
        path_pdrop: float = 0.0,
        mha_win_size: int = -1,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.mha_win_size = mha_win_size
        self.attn = nn.MultiheadAttention(
            embed_dim=n_channels, num_heads=n_head, dropout=attn_pdrop, batch_first=True
        )
        self.attn_drop = nn.Dropout(proj_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Linear(4 * n_channels, n_channels),
            nn.Dropout(proj_pdrop),
        )
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)
        self.drop_path = nn.Dropout(path_pdrop) if path_pdrop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), mask: (B, 1, T)
        b, c, t = x.shape
        xt = x.transpose(1, 2)  # (B, T, C)
        key_padding = ~mask.squeeze(1).bool()  # True at padding

        # Attention
        attn_in = self.norm1(xt)
        if self.mha_win_size is not None and self.mha_win_size > 0:
            # For simplicity keep global attention; original notebook used local windows.
            pass
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding)
        xt = xt + self.drop_path(self.attn_drop(attn_out))

        # MLP
        mlp_in = self.norm2(xt)
        xt = xt + self.drop_path(self.mlp(mlp_in))

        return xt.transpose(1, 2)  # (B, C, T)


class ConvTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        embd_dim: int,
        n_head: int,
        arch: tuple[int, int, int],
        mha_win_size: int,
        scale_factor: int,
        max_seq_len: int,
        attn_pdrop: float,
        proj_pdrop: float,
        path_pdrop: float,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.scale_factor = scale_factor

        # Embedding conv stack
        embed_convs = []
        in_ch = input_dim
        for _ in range(arch[0]):
            embed_convs.append(MaskedConv1D(in_ch, embd_dim, 3, padding=1))
            in_ch = embd_dim
        self.embed_convs = nn.ModuleList(embed_convs)
        self.pos_embd = get_sinusoid_encoding(max_seq_len, embd_dim)

        # Stem transformers (single-scale)
        self.stem = nn.ModuleList(
            [
                TransformerBlock(
                    embd_dim,
                    n_head,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=mha_win_size,
                )
                for _ in range(arch[1])
            ]
        )

        # Branch transformers (multi-scale pyramid)
        self.branches = nn.ModuleList()
        for _ in range(arch[2] + 1):
            self.branches.append(
                TransformerBlock(
                    embd_dim,
                    n_head,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=mha_win_size,
                )
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        for conv in self.embed_convs:
            x, mask = conv(x, mask)

        # Add position encoding (crop/interpolate to length)
        t = x.size(-1)
        if self.pos_embd.size(-1) >= t:
            pe = self.pos_embd[:, :, :t].to(x.device)
        else:
            pe = F.interpolate(self.pos_embd.to(x.device), t, mode="linear", align_corners=False)
        x = x + pe

        for blk in self.stem:
            x = blk(x, mask)

        feats: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        cur_x, cur_m = x, mask
        for lvl, blk in enumerate(self.branches):
            cur_x = blk(cur_x, cur_m)
            feats.append(cur_x)
            masks.append(cur_m)
            if lvl != len(self.branches) - 1:
                # Downsample time by scale_factor
                cur_x = F.avg_pool1d(cur_x, kernel_size=self.scale_factor, stride=self.scale_factor)
                cur_m = F.interpolate(cur_m.float(), size=cur_x.size(-1), mode="nearest").bool()

        return tuple(feats), tuple(masks)


class FPN1D(nn.Module):
    def __init__(self, in_channels: list[int], out_channel: int, scale_factor: int, with_ln: bool = True) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(MaskedConv1D(in_ch, out_channel, 1))
            self.fpn_convs.append(MaskedConv1D(out_channel, out_channel, 3, padding=1, groups=out_channel))
            self.fpn_norms.append(LayerNorm(out_channel) if with_ln else nn.Identity())

    def forward(self, inputs: tuple[torch.Tensor, ...], fpn_masks: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        laterals: list[torch.Tensor] = []
        for lat_conv, feat_mask in zip(self.lateral_convs, zip(inputs, fpn_masks)):
            feat, mask = feat_mask
            lat, _ = lat_conv(feat, mask)
            laterals.append(lat)

        for i in range(len(laterals) - 1, 0, -1):
            target = laterals[i - 1].shape[-1]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=target, mode="nearest")

        fpn_feats: list[torch.Tensor] = []
        new_masks: list[torch.Tensor] = []
        for (lat, mask), conv, norm in zip(zip(laterals, fpn_masks), self.fpn_convs, self.fpn_norms):
            out, new_m = conv(lat, mask)
            out = norm(out)
            fpn_feats.append(out)
            new_masks.append(new_m)
        return tuple(fpn_feats), tuple(new_masks)


class PointGenerator(nn.Module):
    def __init__(self, *, max_seq_len: int, fpn_strides: list[int], regression_range: list[tuple[int, int]]) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.fpn_strides = fpn_strides
        self.regression_range = regression_range

    def forward(self, fpn_feats: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        points: list[torch.Tensor] = []
        for l, feat in enumerate(fpn_feats):
            t = feat.size(-1)
            stride = self.fpn_strides[l]
            reg_min, reg_max = self.regression_range[l]
            pts = torch.arange(0, t, dtype=torch.float32, device=feat.device) * stride + stride / 2
            pts_info = torch.stack(
                [
                    pts,
                    torch.full_like(pts, float(stride)),
                    torch.full_like(pts, float(reg_min)),
                    torch.full_like(pts, float(reg_max)),
                ],
                dim=-1,
            )
            points.append(pts_info)
        return points


class ActionFormer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 256,
        embd_dim: int = 256,
        n_head: int = 4,
        max_seq_len: int = 2048,
        arch: tuple[int, int, int] = (2, 2, 5),
        mha_win_size: int = 19,
        scale_factor: int = 2,
        fpn_dim: int = 256,
        head_dim: int = 256,
        num_classes: int = 1,
        regression_range: list[tuple[int, int]] | None = None,
        attn_pdrop: float = 0.0,
        proj_pdrop: float = 0.0,
        path_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        self.fpn_strides = [scale_factor**i for i in range(arch[2] + 1)]
        if regression_range is None:
            regression_range = [
                (0, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 1024),
            ]
        self.regression_range = regression_range[: len(self.fpn_strides)]

        self.backbone = ConvTransformerBackbone(
            input_dim=input_dim,
            embd_dim=embd_dim,
            n_head=n_head,
            arch=arch,
            mha_win_size=mha_win_size,
            scale_factor=scale_factor,
            max_seq_len=max_seq_len,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
        )
        self.neck = FPN1D(in_channels=[embd_dim] * len(self.fpn_strides), out_channel=fpn_dim, scale_factor=scale_factor)
        self.point_generator = PointGenerator(
            max_seq_len=max_seq_len, fpn_strides=self.fpn_strides, regression_range=self.regression_range
        )

        self.cls_head = nn.ModuleList(
            [nn.Sequential(MaskedConv1D(fpn_dim, head_dim, 3, padding=1), nn.ReLU(), nn.Conv1d(head_dim, num_classes, 1)) for _ in self.fpn_strides]
        )
        self.reg_head = nn.ModuleList(
            [nn.Sequential(MaskedConv1D(fpn_dim, head_dim, 3, padding=1), nn.ReLU(), nn.Conv1d(head_dim, 2, 1)) for _ in self.fpn_strides]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, T, C)
        b, t, _ = x.shape
        if mask is None:
            mask = torch.ones(b, 1, t, device=x.device, dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1)

        feats, masks = self.backbone(x, mask)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        points = self.point_generator(fpn_feats)

        out_cls: list[torch.Tensor] = []
        out_offsets: list[torch.Tensor] = []
        for feat, fpn_mask, cls, reg in zip(fpn_feats, fpn_masks, self.cls_head, self.reg_head):
            # MaskedConv1D in the Sequential expects (x, mask); wrap for compatibility.
            if isinstance(cls[0], MaskedConv1D):
                z, _ = cls[0](feat, fpn_mask)
                z = cls[1](z)
                cls_logits = cls[2](z)
            else:
                cls_logits = cls(feat)

            if isinstance(reg[0], MaskedConv1D):
                z, _ = reg[0](feat, fpn_mask)
                z = reg[1](z)
                offsets = reg[2](z)
            else:
                offsets = reg(feat)
            out_cls.append(cls_logits)
            out_offsets.append(offsets)

        return out_cls, out_offsets, points, fpn_masks


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

