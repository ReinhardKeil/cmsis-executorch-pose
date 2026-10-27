# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Apache-2.0 Google MoveNet Lightning adapter for ExecuTorch/Ethos-U."""

from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from onnx2torch import convert
from PIL import Image, ImageOps
from torch import nn

INPUT_SHAPE = (1, 192, 192, 3)
JOINT_COUNT = 17
MODEL = Path(__file__).with_name("movenet_singlepose_lightning_raw.onnx")
CALIBRATION_IMAGES = Path(__file__).with_name("calibration_pose")


class StaticResize(nn.Module):
    """Fixed 2x resize used by MoveNet's feature-pyramid network."""

    def forward(self, image, _roi, _scales):
        return F.interpolate(
            image, scale_factor=2.0, mode="bilinear", align_corners=False
        )


class MoveNetLightning(nn.Module):
    """Return heatmap, offset, regression, and center heads."""

    def __init__(self):
        super().__init__()
        if not MODEL.exists():
            raise FileNotFoundError(f"{MODEL} is missing")
        self.network = convert(onnx.load(MODEL))
        for name, module in list(self.network.named_modules()):
            if type(module).__name__ == "OnnxResize":
                setattr(self.network, name, StaticResize())

    def forward(self, image):
        heatmaps, offsets, regressions, center = self.network(image)
        return torch.cat((heatmaps, offsets, regressions, center), dim=1)


def get_model(input_shape=INPUT_SHAPE):
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(f"MoveNet Lightning requires {INPUT_SHAPE}")
    return MoveNetLightning().eval()


def get_calibration_inputs(input_shape=INPUT_SHAPE, calibration_samples=8):
    paths = sorted(CALIBRATION_IMAGES.glob("*.jpg"))
    if len(paths) < calibration_samples:
        raise FileNotFoundError(
            f"Need {calibration_samples} pose calibration images in "
            f"{CALIBRATION_IMAGES}; found {len(paths)}"
        )
    inputs = []
    for path in paths[:calibration_samples]:
        with Image.open(path) as source:
            image = ImageOps.pad(
                source.convert("RGB"),
                (input_shape[2], input_shape[1]),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
            )
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        inputs.append(tensor.reshape(input_shape).float() / 127.5 - 1.0)
    return inputs
