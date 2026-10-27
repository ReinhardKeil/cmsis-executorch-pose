# SDS pose test frames

Run `tools/create_pose_sds.py` to letterbox the three photographic fixtures to
192 x 192 RGB888 and create `Pose_In.0.sds`. `Pose_In.sds.yml` describes the
frame contents. In the CMSIS SDS view, use `pose.sdsio.yml` for the DevKit USB
connection or `pose_fvp.sdsio.yml` for the FVP Virtual Streaming Interface.

For a repeatable DevKit run without the VS Code SDS controls, start the server
directly from the repository root after loading and running the firmware:

```shell
sdsio-server.py -c test_images/pose/pose.sdsio.yml --playback 0 \
  --exit-after-playback --no-progress-info --write-flush-records 0
```

The command writes the E7 result to `Pose_Out.0.p.sds` and exits when all three
frames have completed.

`tools/create_pose_sds.py` also writes `Pose_In.transforms.json` with the exact
resize and padding applied to every source photograph. The firmware combines
MoveNet's center, regression, heatmap, and offset heads and reports each refined
coordinate with its heatmap confidence. `tools/view_pose_sds.py` uses the
transform metadata to remove letterbox padding and draw coordinates on the
source photographs.

The fixtures exercise an unobstructed subject, a bent/partly occluded subject,
and a small distant subject. They come from the COCO 2017 validation set:

- `000000000785.jpg`: skier
- `000000000885.jpg`: tennis player, cropped to isolate the primary subject
- `000000001490.jpg`: paddle-boarder

Source URLs use the form
`http://images.cocodataset.org/val2017/<file-name>`. COCO image license ids are
embedded in the validation annotations; the three selected records use license
ids 4, 4, and 2 respectively. See <https://cocodataset.org/#termsofuse>.

These frames validate the complete SDS -> ExecuTorch -> Ethos-U55 -> keypoint
pipeline. The model is the Apache-2.0 Google MoveNet SinglePose Lightning 4
checkpoint. Its convolutional graph is compiled for Ethos-U55; the three
bilinear feature-pyramid resize operations use an ExecuTorch portable kernel.

The eight separate photographs under `model/calibration_pose/` calibrate INT8
activation ranges during AI-layer generation. Evaluation frames are deliberately
excluded from that set; random-noise calibration causes severe loss of accuracy
on the E7 for occluded and small subjects.

To render a recorded pose skeleton over its input frames:

```shell
python tools/view_pose_sds.py test_images/pose/Pose_In.0.sds \
  test_images/pose/Pose_Out.0.p.sds \
  --save test_images/pose/pose_overlay_e7.gif --threshold 0.20
```
