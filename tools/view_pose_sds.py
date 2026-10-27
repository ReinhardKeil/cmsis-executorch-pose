#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Render Pose_Out keypoints over the matching Pose_In SDS frames."""

import argparse
import json
import struct
from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 192
HEIGHT = 192
FRAME_BYTES = WIDTH * HEIGHT * 3
JOINT_COUNT = 17
KEYPOINT_BYTES = JOINT_COUNT * 3 * 4
EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)


def read_sds(path: Path, expected_size: int) -> dict[int, bytes]:
    records = {}
    with path.open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError(f"{path}: truncated SDS record header")
            timeslot, size = struct.unpack("<II", header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError(f"{path}: truncated payload at timeslot {timeslot}")
            if size != expected_size:
                raise ValueError(
                    f"{path}: expected {expected_size} bytes at timeslot {timeslot}, got {size}"
                )
            records[timeslot] = payload
    return records


def render(image: Image.Image, keypoint_data: bytes, threshold: float, scale: int,
           transform: dict | None = None) -> Image.Image:
    keypoints = list(struct.iter_unpack("<fff", keypoint_data))
    draw = ImageDraw.Draw(image)

    def point(index: int) -> tuple[int, int]:
        x, y, _ = keypoints[index]
        input_x = x * (WIDTH - 1)
        input_y = y * (HEIGHT - 1)
        if transform is None:
            return round(input_x), round(input_y)
        source_x = (input_x - transform["pad_left"]) * (
            transform["source_width"] / transform["resized_width"]
        )
        source_y = (input_y - transform["pad_top"]) * (
            transform["source_height"] / transform["resized_height"]
        )
        return (
            round(min(max(source_x, 0), transform["source_width"] - 1)),
            round(min(max(source_y, 0), transform["source_height"] - 1)),
        )

    line_width = max(1, round(image.width / WIDTH))
    radius = max(1, line_width * 2)

    for start, end in EDGES:
        if keypoints[start][2] >= threshold and keypoints[end][2] >= threshold:
            draw.line((point(start), point(end)), fill=(255, 220, 40), width=line_width)
    for index, (_, _, confidence) in enumerate(keypoints):
        if confidence >= threshold:
            x, y = point(index)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 60, 60))

    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Pose_In SDS recording")
    parser.add_argument("output", type=Path, help="Pose_Out SDS or playback recording")
    parser.add_argument("--save", type=Path, default=Path("pose_overlay.gif"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--duration", type=int, default=500, help="GIF frame duration in ms")
    parser.add_argument(
        "--transforms", type=Path,
        help="Letterbox metadata; defaults to Pose_In.transforms.json beside the input",
    )
    args = parser.parse_args()

    input_records = read_sds(args.input, FRAME_BYTES)
    output_records = read_sds(args.output, KEYPOINT_BYTES)
    timeslots = sorted(input_records.keys() & output_records.keys())
    if not timeslots:
        raise SystemExit("No matching input and output timeslots found")

    transform_path = args.transforms or args.input.with_name("Pose_In.transforms.json")
    transforms = None
    if transform_path.exists():
        metadata = json.loads(transform_path.read_text())
        transforms = {frame["timeslot"]: frame for frame in metadata["frames"]}

    frames = []
    for timeslot in timeslots:
        transform = transforms.get(timeslot) if transforms else None
        if transform:
            source_path = transform_path.parents[2] / transform["source"]
            image = Image.open(source_path).convert("RGB")
        else:
            image = Image.frombytes("RGB", (WIDTH, HEIGHT), input_records[timeslot])
        frames.append(render(image, output_records[timeslot], args.threshold, args.scale, transform))

    max_width = max(frame.width for frame in frames)
    max_height = max(frame.height for frame in frames)
    padded_frames = []
    for frame in frames:
        canvas = Image.new("RGB", (max_width, max_height), "black")
        canvas.paste(frame, (0, 0))
        padded_frames.append(canvas)
    frames = padded_frames
    args.save.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.save,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )
    print(f"Wrote {len(frames)} frames to {args.save}")


if __name__ == "__main__":
    main()
