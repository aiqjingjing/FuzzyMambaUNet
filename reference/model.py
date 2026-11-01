import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .primitives import (
    EVLA,
    SS2D,
    DropPath,
    Final_PatchExpand2D,
    FusedPosEMA,
    PatchEmbed2D,
    PatchExpand2D,
    PatchMerging2D,
    trunc_normal_,
)


class PATE(FusedPosEMA):
    def __init__(self, channels, groups=16):
        if groups < 1 or channels % groups:
            raise ValueError("PATE groups must divide every stage channel count")
        super().__init__(channels, factor=groups)


class EVGLA(EVLA):
    def __init__(self, channels, global_branch="variance"):
        super().__init__(channels)
        if global_branch not in ("variance", "plain", "ssm"):
            raise ValueError("Invalid global branch")
        self.global_branch = global_branch
        if global_branch == "ssm":
            self.global_norm = nn.LayerNorm(channels)
            self.global_ssm = SS2D(d_model=channels, d_state=16, dropout=0.0)

    def forward(self, x):
        z = x.permute(0, 3, 1, 2)
        if self.global_branch == "ssm":
            global_feature = self.global_ssm(self.global_norm(x)).permute(0, 3, 1, 2)
        else:
            global_feature = self.ca(z) * z
            global_feature = self.sa(global_feature) * global_feature
            if self.global_branch == "variance":
                global_feature = self.spatialvariance(global_feature)
        rows = []
        for row in z.chunk(2, dim=2):
            quadrants = []
            for quadrant in row.chunk(2, dim=3):
                quadrant = self.ca(quadrant) * quadrant
                quadrants.append(self.sa(quadrant) * quadrant)
            rows.append(torch.cat(quadrants, dim=3))
        local_feature = self.enhancer(torch.cat(rows, dim=2))
        return (global_feature + local_feature).permute(0, 2, 3, 1)


class GaussianProduct(nn.Module):
    def __init__(
        self, channels, concepts=4, width=1.0, center_std=0.1, operator="product"
    ):
        super().__init__()
        if concepts < 1 or width <= 1e-4 or center_std < 0:
            raise ValueError("Invalid fuzzy initialization")
        if operator not in ("product", "minimum", "mean"):
            raise ValueError("Unknown operator")
        self.operator = operator
        self.centers = nn.Parameter(torch.randn(channels, concepts) * center_std)
        self.raw_widths = nn.Parameter(
            torch.full((channels, concepts), math.log(math.expm1(width - 1e-4)))
        )

    def memberships(self, residual):
        widths = F.softplus(self.raw_widths) + 1e-4
        delta = residual.unsqueeze(2) - self.centers[None, :, :, None, None]
        log_mu = -(delta / widths[None, :, :, None, None]).square()
        return log_mu

    def forward(self, residual):
        log_mu = self.memberships(residual)
        if self.operator == "product":
            return log_mu.sum(dim=2).exp()
        mu = log_mu.exp()
        return mu.min(dim=2).values if self.operator == "minimum" else mu.mean(dim=2)


class FA(nn.Module):
    def __init__(
        self,
        channels,
        concepts=4,
        operator="product",
        gate="fuzzy",
        width=1.0,
        center_std=0.1,
    ):
        super().__init__()
        if gate not in ("fuzzy", "sigmoid", "identity"):
            raise ValueError("Unknown residual gate")
        self.gate = gate
        self.channel_decomposer = nn.Conv2d(channels, 1, 1)
        self.fuzzy_decomposer = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
        )
        self.principal_center = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.principal_raw_width = nn.Parameter(
            torch.full((1, 1, 1, 1), math.log(math.expm1(0.5 - 1e-4)))
        )
        if gate == "fuzzy":
            self.residual_gate = GaussianProduct(
                channels, concepts, width, center_std, operator
            )
        elif gate == "sigmoid":
            self.residual_gate = nn.Sequential(
                nn.Conv2d(channels, channels, 1, groups=channels), nn.Sigmoid()
            )
        else:
            self.residual_gate = None
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-5))
        self.capture = False
        self.last_maps = None

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        principal = F.gelu(self.channel_decomposer(x))
        width = F.softplus(self.principal_raw_width) + 1e-4
        principal_mu = (
            -((self.fuzzy_decomposer(x) - self.principal_center) / width).square()
        ).exp()
        residual = x - (0.5 * principal + 0.5 * principal_mu)
        rho = (
            self.residual_gate(residual)
            if self.residual_gate is not None
            else torch.ones_like(residual)
        )
        if self.capture:
            maps = dict(
                principal_membership=principal_mu.detach().cpu(),
                residual_reliability=rho.mean(1, keepdim=True).detach().cpu(),
                residual_magnitude=residual.abs().mean(1, keepdim=True).detach().cpu(),
            )
            if self.gate == "fuzzy":
                maps["concept_memberships"] = (
                    self.residual_gate.memberships(residual)
                    .exp()
                    .mean(1)
                    .detach()
                    .cpu()
                )
            self.last_maps = maps
        return (x + self.gamma * rho * residual).permute(0, 2, 3, 1)


class FuzzySkipEnhancer(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.ev = EVGLA(channels, config["global_branch"]) if config["evgla"] else None
        self.adaptive_weight = (
            nn.Conv2d(channels, 1, 1) if self.ev is not None else None
        )
        self.fa = (
            FA(
                channels,
                config["concepts"],
                config["operator"],
                config["gate"],
                config["fuzzy_width"],
                config["center_std"],
            )
            if config["fa"]
            else None
        )
        if self.fa is not None:
            self.beta = nn.Parameter(torch.tensor(0.15))

    def forward(self, x):
        if self.ev is not None:
            weight = (
                self.adaptive_weight(x.permute(0, 3, 1, 2))
                .sigmoid()
                .permute(0, 2, 3, 1)
            )
            x = x + weight * (self.ev(x) - x)
        if self.fa is not None:
            x = x + self.beta.sigmoid() * (self.fa(x) - x)
        return x


class PTAMambaBlock(nn.Module):
    def __init__(self, dim, drop_path, pate, groups):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attention = SS2D(d_model=dim, d_state=16, dropout=0.0)
        self.drop_path = DropPath(drop_path)
        self.pate = PATE(dim, groups) if pate else None

    def forward(self, x):
        y = x + self.drop_path(self.self_attention(self.ln_1(x)))
        return y + self.pate(x) if self.pate is not None else y


class Stage(nn.Module):
    def __init__(self, dim, rates, config, downsample=False, upsample=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PTAMambaBlock(dim, r, config["pate"], config["groups"]) for r in rates]
        )
        self.downsample = PatchMerging2D(dim) if downsample else None
        self.upsample = PatchExpand2D(dim) if upsample else None

    def blocks_forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


DEFAULT_MODEL = dict(
    pate=True,
    evgla=True,
    fa=True,
    concepts=4,
    operator="product",
    gate="fuzzy",
    fuzzy_width=1.0,
    center_std=0.1,
    groups=16,
    drop_path=0.2,
    global_branch="variance",
    mc_dropout_p=0.0,
)


class FuzzyMambaUNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(DEFAULT_MODEL, **(config or {}))
        dims = [96, 192, 384, 768]
        rates = torch.linspace(0, self.config["drop_path"], 8).tolist()
        reverse_rates = list(reversed(rates))
        self.patch_embed = PatchEmbed2D(4, 3, 96, norm_layer=nn.LayerNorm)
        self.layers = nn.ModuleList(
            [
                Stage(d, rates[2 * i : 2 * i + 2], self.config, downsample=i < 3)
                for i, d in enumerate(dims)
            ]
        )
        self.skip_enhancers = nn.ModuleList(
            [FuzzySkipEnhancer(d, self.config) for d in dims[:3]]
        )
        self.layers_up = nn.ModuleList(
            [
                Stage(d, reverse_rates[2 * i : 2 * i + 2], self.config, upsample=i > 0)
                for i, d in enumerate(reversed(dims))
            ]
        )
        self.final_up = Final_PatchExpand2D(96, dim_scale=4)
        self.final_conv = nn.Conv2d(24, 1, 1)
        self.mc_dropout = nn.Dropout2d(self.config["mc_dropout_p"])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] % 32 or x.shape[3] % 32:
            raise ValueError("Expected Bx3xHxW with H,W divisible by 32")
        if max(x.shape[2:]) // 4 > 512:
            raise ValueError(
                "PATE positional storage supports feature dimensions up to 512"
            )
        x = self.patch_embed(x)
        skips = []
        for i, stage in enumerate(self.layers):
            x = stage.blocks_forward(x)
            if i < 3:
                skips.append(self.skip_enhancers[i](x))
                x = stage.downsample(x)
        for i, stage in enumerate(self.layers_up):
            if i:
                x = stage.upsample(x)
                skip = skips[3 - i]
                if x.shape != skip.shape:
                    raise RuntimeError("Decoder and skip feature shapes disagree")
                x = x + skip
            x = stage.blocks_forward(x)
        return self.final_conv(
            self.mc_dropout(self.final_up(x).permute(0, 3, 1, 2))
        ).sigmoid()

    def load_pretrained(self, path, mirror_decoder=True):
        path = Path(path)
        raw = torch.load(str(path), map_location="cpu")
        source = raw.get("model", raw)
        target = self.state_dict()
        loaded, ignored, mismatched = {}, [], []
        for key, value in source.items():
            candidates = [key]
            parts = key.split(".")
            if (
                mirror_decoder
                and len(parts) > 3
                and parts[0] == "layers"
                and parts[2] == "blocks"
            ):
                candidates.append(
                    "layers_up." + str(3 - int(parts[1])) + "." + ".".join(parts[2:])
                )
            for dest in candidates:
                if dest not in target:
                    ignored.append(dest)
                elif target[dest].shape != value.shape:
                    mismatched.append(dest)
                else:
                    loaded[dest] = value
        if "patch_embed.proj.weight" not in loaded or not any(
            ".self_attention." in k for k in loaded
        ):
            raise RuntimeError("Checkpoint did not match the expected VMamba backbone")
        self.load_state_dict(loaded, strict=False)
        return dict(
            path=str(path.resolve()),
            loaded_tensors=len(loaded),
            loaded_parameter_elements=sum(
                v.numel()
                for k, v in loaded.items()
                if k in dict(self.named_parameters())
            ),
            encoder_tensors=sum(k.startswith("layers.") for k in loaded),
            decoder_tensors=sum(k.startswith("layers_up.") for k in loaded),
            shape_mismatches=mismatched,
            unused_source_targets=ignored,
            newly_initialized_keys=[k for k in target if k not in loaded],
        )

    def capture_fuzzy(self, enabled=True):
        for module in self.modules():
            if isinstance(module, FA):
                module.capture = enabled
                module.last_maps = None
