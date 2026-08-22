"""Advisor-authorized ECF+BOSR three-seed heads.

This module contains no dataset or evaluation path.  BOSR-only is an operation
ablation of the same 5-output parameterization: alpha and beta are forced to
zero while the odds-space update remains active.
"""
from __future__ import annotations

import torch
from torch import nn


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.act(self.dw(x)))


class FormalBOSRHead(nn.Module):
    GROUPS = ("BOSR-ONLY", "ECF+BOSR")

    def __init__(self, group: str, width: int = 24, feedback_bound: float = 0.10) -> None:
        super().__init__()
        if group not in self.GROUPS:
            raise ValueError(f"unsupported group: {group}")
        self.group = group
        self.feedback_bound = float(feedback_bound)
        self.stem = nn.Sequential(nn.Conv2d(15, width, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(DepthwiseResidualBlock(width), DepthwiseResidualBlock(width))
        self.output = nn.Conv2d(width, 5, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, i0: torch.Tensor, ic: torch.Tensor, i1: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dc, db = ic - i0, i1 - ic
        features = torch.cat((i0, ic, i1, dc, db), dim=1)
        logits = self.output(self.body(self.stem(features)))
        coefficients = self.feedback_bound * torch.tanh(logits[:, :2])
        alpha, beta = coefficients[:, 0:1], coefficients[:, 1:2]
        if self.group == "BOSR-ONLY":
            alpha, beta = torch.zeros_like(alpha), torch.zeros_like(beta)
        stage_preclip = i1 + alpha * dc + beta * db
        i_ecf = torch.clamp(stage_preclip, 0.0, 1.0)
        u = torch.tanh(logits[:, 2:])
        exp_u = torch.exp(u)
        output = i_ecf * exp_u / (1.0 - i_ecf + i_ecf * exp_u)
        return output, {
            "alpha": alpha,
            "beta": beta,
            "u": u,
            "i_ecf": i_ecf,
            "stage_preclip": stage_preclip,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

