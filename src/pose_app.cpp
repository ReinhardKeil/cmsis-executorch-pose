// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "cmsis_os2.h"
#include "sds.h"
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "model_pte.h"

namespace {
constexpr size_t kWidth = 192;
constexpr size_t kHeight = 192;
constexpr size_t kChannels = 3;
constexpr size_t kFrameBytes = kWidth * kHeight * kChannels;
constexpr size_t kJoints = 17;
constexpr size_t kHeatWidth = 48;
constexpr size_t kHeatHeight = 48;
constexpr size_t kHeatmapChannels = 17;
constexpr size_t kOffsetChannels = 34;
constexpr size_t kRegressionChannels = 34;
constexpr float kDecodeRegularizer = 1.7998047F;
constexpr size_t kMethodPoolSize = 2U * 1024U * 1024U;
constexpr size_t kTempPoolSize = 2U * 1024U * 1024U;

alignas(16) uint8_t method_pool[kMethodPoolSize];
#if defined(APP_DEVKIT_E7)
alignas(16) uint8_t temp_pool[kTempPoolSize] __attribute__((section(".bss.sram1")));
#else
alignas(16) uint8_t temp_pool[kTempPoolSize];
#endif
alignas(16) uint8_t frame[kFrameBytes];
alignas(16) float input[kFrameBytes];
alignas(4) float keypoints[kJoints * 3];
alignas(4) uint8_t sds_in_buffer[kFrameBytes * 2 + 2048];
alignas(4) uint8_t sds_out_buffer[sizeof(keypoints) * 2 + 2048];

sdsId_t input_stream = nullptr;
sdsId_t output_stream = nullptr;

void sds_event(sdsId_t, uint32_t event) {
  if ((event & SDS_EVENT_ERROR_IO) != 0U) {
    std::printf("SDS I/O error\n");
  }
}

void decode_heatmaps(const executorch::aten::Tensor& tensor) {
  const float* heads = tensor.const_data_ptr<float>();
  constexpr size_t plane = kHeatWidth * kHeatHeight;
  const float* heatmaps = heads;
  const float* offsets = heatmaps + kHeatmapChannels * plane;
  const float* regressions = offsets + kOffsetChannels * plane;
  const float* center = regressions + kRegressionChannels * plane;

  size_t center_index = 0;
  float center_score = -1.0F;
  for (size_t y = 0; y < kHeatHeight; ++y) {
    for (size_t x = 0; x < kHeatWidth; ++x) {
      const float dy = static_cast<float>(y) - kHeatHeight / 2.0F;
      const float dx = static_cast<float>(x) - kHeatWidth / 2.0F;
      const float score = center[y * kHeatWidth + x] /
                          (std::sqrt(dx * dx + dy * dy) + kDecodeRegularizer);
      if (score > center_score) {
        center_score = score;
        center_index = y * kHeatWidth + x;
      }
    }
  }
  const float center_y = static_cast<float>(center_index / kHeatWidth);
  const float center_x = static_cast<float>(center_index % kHeatWidth);

  for (size_t joint = 0; joint < kJoints; ++joint) {
    const float* heatmap = heatmaps + joint * kHeatWidth * kHeatHeight;
    const float predicted_y =
        center_y + regressions[(joint * 2) * plane + center_index];
    const float predicted_x =
        center_x + regressions[(joint * 2 + 1) * plane + center_index];
    size_t best = 0;
    float best_score = -1.0F;
    for (size_t y = 0; y < kHeatHeight; ++y) {
      for (size_t x = 0; x < kHeatWidth; ++x) {
        const float dy = static_cast<float>(y) - predicted_y;
        const float dx = static_cast<float>(x) - predicted_x;
        const size_t index = y * kHeatWidth + x;
        const float score = heatmap[index] /
                            (std::sqrt(dx * dx + dy * dy) + kDecodeRegularizer);
        if (score > best_score) {
          best_score = score;
          best = index;
        }
      }
    }
    const float refined_y = static_cast<float>(best / kHeatWidth) +
                            offsets[(joint * 2) * plane + best];
    const float refined_x = static_cast<float>(best % kHeatWidth) +
                            offsets[(joint * 2 + 1) * plane + best];
    keypoints[joint * 3 + 0] = refined_x / kHeatWidth;
    keypoints[joint * 3 + 1] = refined_y / kHeatHeight;
    keypoints[joint * 3 + 2] = heatmap[best];
  }
}

void pose_thread(void*) {
  if (sdsInit(sds_event) != SDS_OK) {
    std::printf("SDS initialization failed\n");
    osThreadExit();
  }

  executorch::runtime::runtime_init();
  arm::embedded::EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<executorch::extension::BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<executorch::runtime::MemoryAllocator>(kMethodPoolSize, method_pool),
      std::make_unique<executorch::runtime::MemoryAllocator>(kTempPoolSize, temp_pool));

  std::array<executorch::aten::SizesType, 4> sizes{1, kHeight, kWidth, kChannels};
  std::array<executorch::aten::DimOrderType, 4> order{0, 1, 2, 3};
  executorch::aten::TensorImpl impl(executorch::aten::ScalarType::Float, sizes.size(),
                                    sizes.data(), input, order.data());
  executorch::aten::Tensor input_tensor(&impl);
  std::printf("Pose estimator ready: %lu-byte ExecuTorch model; waiting for SDS Pose_In\n",
              model_pte_size);

  for (;;) {
    sdsExchange();
    if ((sdsFlags & SDS_FLAG_TERMINATE) != 0U) {
      std::putchar(4);
      osThreadExit();
    }
    if ((sdsFlags & SDS_FLAG_ALIVE) == 0U) {
      osDelay(20);
      continue;
    }
    if ((sdsFlags & SDS_FLAG_START) == 0U) {
      osDelay(20);
      continue;
    }

    input_stream = sdsOpen("Pose_In", sdsModeRead, sds_in_buffer, sizeof(sds_in_buffer));
    output_stream = sdsOpen("Pose_Out", sdsModeWrite, sds_out_buffer, sizeof(sds_out_buffer));
    if (input_stream == nullptr || output_stream == nullptr) {
      std::printf("Unable to open SDS pose streams\n");
      osDelay(100);
      continue;
    }

    uint32_t timeslot = 0;
    for (;;) {
      int32_t count = sdsRead(input_stream, &timeslot, frame, sizeof(frame));
      if (count == SDS_NO_DATA) {
        osDelay(1);
        continue;
      }
      if (count <= 0 || static_cast<size_t>(count) != sizeof(frame)) break;
      for (size_t i = 0; i < kFrameBytes; ++i) input[i] = frame[i] / 127.5F - 1.0F;

      auto outputs = module.forward(input_tensor);
      if (!outputs.ok()) {
        std::printf("Pose inference failed: ExecuTorch error 0x%02x\n",
                    static_cast<unsigned>(outputs.error()));
        break;
      }
      if (outputs->empty() || !outputs->front().isTensor()) {
        std::printf("Pose inference returned no tensor\n");
        break;
      }
      decode_heatmaps(outputs->front().toTensor());
      while (sdsWrite(output_stream, timeslot, keypoints, sizeof(keypoints)) == SDS_NO_SPACE) {
        osDelay(1);
      }
      std::printf("pose @%u: nose=(%.3f, %.3f)\n", timeslot, keypoints[0], keypoints[1]);
    }
    sdsClose(input_stream);
    sdsClose(output_stream);
    input_stream = nullptr;
    output_stream = nullptr;
    sdsFlagsModify(0U, SDS_FLAG_START);
  }
}
}  // namespace

extern "C" int app_main(void) {
  osKernelInitialize();
  osThreadAttr_t attributes{};
  attributes.name = "Pose";
  attributes.stack_size = 8192;
  if (osThreadNew(pose_thread, nullptr, &attributes) == nullptr) {
    std::printf("Pose thread creation failed\n");
    return 1;
  }
  osKernelStart();
  return 0;
}
