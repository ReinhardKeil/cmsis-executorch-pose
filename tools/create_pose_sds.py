#!/usr/bin/env python3
"""Letterbox representative COCO pose images into an SDS recording."""

import json
from pathlib import Path
import struct
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "test_images" / "pose"
OUTPUT = FRAMES / "Pose_In.0.sds"
TRANSFORMS = FRAMES / "Pose_In.transforms.json"
SOURCES = (
    FRAMES / "coco" / "coco_000000000785_ski.jpg",
    FRAMES / "coco" / "coco_000000000885_tennis_crop.jpg",
    FRAMES / "coco" / "coco_000000001490_surf.jpg",
)

transforms = []
with OUTPUT.open("wb") as stream:
    for index, path in enumerate(SOURCES):
        if not path.exists():
            raise FileNotFoundError(f"Missing pose fixture: {path}")
        with Image.open(path) as source:
            source = source.convert("RGB")
            source_width, source_height = source.size
            scale = min(192 / source_width, 192 / source_height)
            resized_width = round(source_width * scale)
            resized_height = round(source_height * scale)
            resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
            pad_left = (192 - resized_width) // 2
            pad_top = (192 - resized_height) // 2
            image = Image.new("RGB", (192, 192))
            image.paste(resized, (pad_left, pad_top))
        payload = image.tobytes()
        timeslot = index * 500_000
        stream.write(struct.pack("<II", timeslot, len(payload)))
        stream.write(payload)
        transforms.append({
            "timeslot": timeslot,
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_width": source_width,
            "source_height": source_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "pad_left": pad_left,
            "pad_top": pad_top,
        })

TRANSFORMS.write_text(json.dumps({"input_width": 192, "input_height": 192,
                                  "frames": transforms}, indent=2) + "\n")

print(f"Wrote {len(SOURCES)} COCO pose frames to {OUTPUT} and {TRANSFORMS}")
