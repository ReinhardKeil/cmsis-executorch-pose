# Using ExecuTorch with CMSIS Developer Assistent

This project started from the
[Arm-Examples/CMSIS-Executorch](https://github.com/Arm-Examples/CMSIS-Executorch)
template repository. That template supplied the TinyCNN examples, CMSIS
solutions, ExecuTorch integration, and MLOps flow for Corstone FVPs.

Using the CMSIS Developer Assistant, the template was extended into a
17-keypoint pose application for an Alif Ensemble DevKit-E7 with
Ethos-U55-256. The result covers model conversion and quantization, target
integration, building, flashing, SDS test-data streaming, hardware validation,
and visualization.

![Alif E7 pose-estimation results](test_images/pose/pose_overlay_e7.gif)

## What was built

The project demonstrates the complete embedded ML path:

1. Describe the processor, NPU, Vela memory, and model input in CMSIS.
2. Export and quantize a PyTorch model with ExecuTorch.
3. Generate a target-specific CMSIS AI layer.
4. Build and run on an FVP or physical board.
5. Stream representative test data with SDS and validate the output.

| Solution | Application | Target |
| --- | --- | --- |
| `cmsis-executorch.csolution.yml` | TinyCNN | Corstone-320 / Ethos-U85-256 FVP |
| `cmsis-executorch-u55.csolution.yml` | TinyCNN | Corstone-300 / Ethos-U55 FVP |
| `cmsis-executorch-pose.csolution.yml` | Google MoveNet Lightning | Corstone-300 FVP and Alif DevKit-E7 / M55_HP / Ethos-U55-256 |

## Prerequisites

- Git and Keil Studio for VS Code.
- CMSIS-Toolbox 2.15 or newer.
- Python `>=3.10,<3.14`, or `uv` for Python setup.
- Keil MDK Community or another suitable Arm Compiler license.
- An Alif DevKit-E7 and debug probe for hardware validation.

On Windows, enable
[long-path support](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation#enable-long-paths-in-windows-10-version-1607-and-later).

## Quick start

Create the Python environment:

```powershell
.\setup_venv.bat --uv --python 3.12
```

On Linux or macOS, use `./setup_venv.sh --uv --python 3.12`. The environment
pins `executorch==1.5.1` to match `PyTorch::ExecuTorch@1.5.1`.

Open the repository in Keil Studio, select a `*.csolution.yml` in the CMSIS view,
then use the CMSIS **Build**, **Run**, or **Debug** actions. The generated AI
layers are committed, so the solutions build without first running Python.

Command-line examples:

```shell
cbuild cmsis-executorch.csolution.yml --active SSE-320-U85
cbuild cmsis-executorch-pose.csolution.yml --active DevKit-E7-HP
```

After changing a model or target, regenerate its AI layer:

```shell
cbuild setup cmsis-executorch-pose.csolution.yml --active DevKit-E7-HP --packs
python create_ai_layer.py cmsis-executorch-pose.cbuild-mlops.yml
cbuild cmsis-executorch-pose.csolution.yml --active DevKit-E7-HP
```

On Windows, use `.\.venv\Scripts\python.exe`. The export reads the processor,
NPU, and Vela settings from `*.cbuild-mlops.yml`; target configuration is not
duplicated in the model code.

## Creating the pose application with CMSIS Developer Assistant

Starting from the
[CMSIS-Executorch template](https://github.com/Arm-Examples/CMSIS-Executorch)
with the Alif DevKit-E7 connected, these three prompts capture the essential
workflow. The resulting implementation is
described in [From template to an Alif E7 pose application](#from-template-to-an-alif-e7-pose-application).

1. **Create the application**

   > Use this ExecuTorch template to create a 17-keypoint pose-estimation
   > application from the Apache-2.0 Google MoveNet Lightning model. Target the Alif
   > DevKit-E7 M55_HP and Ethos-U55-256, retain the CMSIS MLOps flow, compile
   > the convolutional workload for the NPU, and preserve the model's bilinear
   > feature-pyramid resize.

2. **Add data and validate the hardware**

   > Add separate representative photographic data for INT8 calibration and
   > evaluation. Stream 192 x 192 RGB frames and pose results over USB SDS,
   > build and flash the application, then validate it on the connected Alif E7
   > using the SDSIO Server directly.

3. **Improve and document the coordinates**

   > Add MoveNet center, regression, heatmap, offset, and confidence decoding. Preserve the input
   > letterbox transform, map results back to the original photographs, generate
   > an animated result GIF, and document how to reproduce the test.

The CMSIS Developer Assistant checks the active target and session before using
CMSIS actions to build, flash, run, and inspect the board.

## From template to an Alif E7 pose application

The upstream CMSIS-Executorch template already provided the CMSIS solution
structure, ExecuTorch runtime integration, AI-layer generation, and TinyCNN FVP
examples. This work retained that architecture and added the Alif E7 target,
pose model and firmware, representative image data, SDS transport, coordinate
reconstruction, board validation, and result visualization.

### Model and quantization

`model/movenet_singlepose_lightning_raw.onnx` contains the backbone and four raw
heads extracted from Google's Apache-2.0 MoveNet Lightning float16 model. The
adapter in `model/pose_model.py` imports that graph into PyTorch, retains the
model's bilinear feature-pyramid resize, and exposes the heads to ExecuTorch.
Vela compiles the convolutional graph into four Ethos-U regions; three small
bilinear resize operations use the ExecuTorch portable kernel.

Eight photographs in `model/calibration_pose/` provide representative INT8
activation ranges. Random-noise calibration caused large coordinate errors on
occluded and distant people; photographic calibration corrected the hardware
results. `create_ai_layer.py` exports the quantized program into
`ai_layer_pose/`.

### Firmware and coordinates

`src/pose_app.cpp` receives 192 x 192 RGB888 frames over USB SDS, normalizes
them to `[-1, 1]`, and runs ExecuTorch. It decodes MoveNet's center heatmap,
keypoint regressions, per-keypoint heatmaps, and offsets into 17 normalized
coordinates with confidence values.

`tools/create_pose_sds.py` letterboxes three COCO validation photographs into
`Pose_In.0.sds` and records each resize and padding transform in
`Pose_In.transforms.json`.

After loading the firmware on the E7, run:

```shell
sdsio-server.py -c test_images/pose/pose.sdsio.yml --playback 0 \
  --exit-after-playback --no-progress-info --write-flush-records 0
```

Render the recorded board output on the original photographs:

```shell
python tools/view_pose_sds.py \
  test_images/pose/Pose_In.0.sds \
  test_images/pose/Pose_Out.0.p.sds \
  --save test_images/pose/pose_overlay_e7.gif --threshold 0.20
```

The viewer reverses the letterbox transformation before drawing the skeleton.
See [`test_images/pose/README.md`](test_images/pose/README.md) for data
provenance and file-format details.

## MLOps architecture

```text
*.csolution.yml
       | cbuild setup
       v
*.cbuild-mlops.yml  --->  create_ai_layer.py  --->  ai_layer*/
       |                                             |
       +---------------- cbuild ---------------------+
                             |
                             v
                    FVP or physical board
                             |
                             v
                      SDS validation data
```

CMSIS describes the hardware, Python produces a target-specific AI layer, and
the regular CMSIS build consumes that layer without invoking Python.

## Project layout

| Path | Purpose |
| --- | --- |
| `cmsis-executorch*.csolution.yml` | Target and MLOps configurations |
| `model/` | TinyCNN, the Apache-2.0 MoveNet ONNX checkpoint, adapters, and calibration data |
| `create_ai_layer.py` | Export, quantization, delegation, and AI-layer generation |
| `ai_layer/`, `ai_layer_pose/` | Generated target-specific AI layers |
| `src/` | ExecuTorch applications and embedded module wrapper |
| `board/` | Corstone and Alif board integrations |
| `sdsio/` | FVP, USB, and RTT SDS transports |
| `test_images/pose/` | Pose input, E7 output, transforms, and visualization |
| `documentation/` | Detailed MLOps and pack-maintenance notes |

## Adapting the project

To deploy another model:

1. Implement `get_model()` and `get_calibration_inputs()`.
2. Update the solution's `mlops.model` settings.
3. Regenerate the `*.cbuild-mlops.yml` and AI layer.
4. Update application input/output handling.
5. Validate with representative data on the intended target.

For camera models, prefer NHWC input so the first NPU convolution can absorb
the layout conversion.

## Further documentation

- [MLOps flow](documentation/mlops-flow.md)
- [ExecuTorch pack maintenance](documentation/executorch-pack-maintenance.md)
- [PyTorch ExecuTorch pack](https://www.keil.arm.com/packs/executorch-pytorch/)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [CMSIS-Ethos-U](https://arm-software.github.io/CMSIS-Ethos-U/)
- [CMSIS-Toolbox MLOps](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)

## License

The template code is licensed under Apache-2.0; see [LICENSE](LICENSE).
`src/arm_embedded_module.*`, derived from ExecuTorch, uses BSD-3-Clause; see
[`src/LICENSE-ExecuTorch`](src/LICENSE-ExecuTorch).

The pose checkpoint is derived from
[Google MoveNet SinglePose Lightning 4](https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4),
which is published under Apache-2.0. The COCO validation images retain their
individual source licenses; see [`test_images/pose/README.md`](test_images/pose/README.md).
