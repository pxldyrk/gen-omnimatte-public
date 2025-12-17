"""Omnimatte optimization and utilities."""

from omnimatte.optimization import OmnimatteOptimizer
from omnimatte.utils import save_omnimatte, refine_mask, transfer_detail
from omnimatte.loss import OmnimatteLoss
from omnimatte.modules import OmnimatteNet
from omnimatte.animation import OmnimatteAnimation

__all__ = [
    "OmnimatteOptimizer",
    "OmnimatteLoss",
    "OmnimatteNet",
    "OmnimatteAnimation",
    "save_omnimatte",
    "refine_mask",
    "transfer_detail",
]
