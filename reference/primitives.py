import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath as DropPath
from timm.models.layers import trunc_normal_ as trunc_normal_
from torch import nn

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as error:
    raise ImportError(
        "Install mamba-ssm with a matching PyTorch/CUDA environment."
    ) from error


class PatchEmbed2D(nn.Module):
    def __init__(
        self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(
                f"Warning, x.shape {x.shape} is not match even ===========", flush=True
            )
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x1 = x1[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x2 = x2[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]
            x3 = x3[:, : SHAPE_FIX[0], : SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // self.dim_scale,
        )
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        if dts.stride(-1) != 1:
            dts = torch.empty(dts.shape, device=dts.device, dtype=dts.dtype).copy_(dts)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FusedPosEMA(nn.Module):
    def __init__(self, channels, max_len=512, factor=16):
        super().__init__()

        self.pos_h = nn.Parameter(torch.randn(1, channels, max_len, 1) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(1, channels, 1, max_len) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.groups = factor
        assert channels // self.groups > 0

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3x3 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, c, h, w = x.size()

        pos_h = self.pos_h[:, :, :h, :].expand(b, c, h, w)
        pos_w = self.pos_w[:, :, :, :w].expand(b, c, h, w)
        pos_encoding = torch.sigmoid(self.alpha) * (pos_h + pos_w)
        x = x + pos_encoding

        group_x = x.reshape(b * self.groups, -1, h, w)

        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        )
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        output = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        return output.permute(0, 2, 3, 1)


class EVLA(nn.Module):
    def __init__(self, in_channel):
        super(EVLA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()
        self.enhancer = EdgeAwareFeatureEnhancer(in_channels=in_channel)
        self.spatialvariance = SpatialVarianceModulation()

    def forward(self, x):
        x_permuted = x.permute(0, 3, 1, 2)

        x4 = self.ca(x_permuted) * x_permuted
        x4 = self.sa(x4) * x4

        x0, x1 = x_permuted.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)

        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x3 = self.enhancer(x3)
        x4 = self.spatialvariance(x4)

        result = x3 + x4

        return result.permute(0, 2, 3, 1)


class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAwareFeatureEnhancer, self).__init__()

        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        edge_features = x - self.edge_extractor(x)

        edge_weights = self.weight_generator(edge_features)

        enhanced_features = edge_weights * x + x
        return enhanced_features


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialVarianceModulation(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super(SpatialVarianceModulation, self).__init__()
        self.activation = nn.Sigmoid()
        self.eps = eps

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.size()

        spatial_mean = feature_map.mean(dim=[2, 3], keepdim=True)

        squared_deviation = (feature_map - spatial_mean).pow(2)

        spatial_variance = squared_deviation.sum(dim=[2, 3], keepdim=True) / (
            height * width - 1
        )

        modulation_coeff = squared_deviation / (4 * (spatial_variance + self.eps)) + 0.5
        weight = self.activation(modulation_coeff)

        modulated_feature = feature_map * weight
        return modulated_feature
