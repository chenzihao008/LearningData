/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_fp16.h>
#include <numeric>

#include "camera-bevpool.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

namespace bevfusion {
namespace camera {

#define tile_size 10

typedef struct {
  unsigned int val[5];
} combined_half;

static __global__ void bevpool_half_pack10_kernel(const half* camera_feature, const half* depth_weights, unsigned int nchannel,
                                                  const int3* intervals, unsigned int n_intervals, const unsigned int* indices,
                                                  unsigned int out_h, unsigned int out_w, unsigned int ndepth, unsigned int farea,
                                                  half* output_bevfeat) {
  int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
  int feature_block = threadIdx.x * tile_size;

  if (interval_index >= n_intervals) return;
  int3 interval = intervals[interval_index];
  half accumulate[tile_size] = {0.};

  // 对每一个interval, 将属于这个interval中的所有的点的每一个feature进行accumulate
  // 注意，从interval寻找对应的camera_feature_offset的过程其实是没有涉及计算的，单纯的寻址
  for (int i = interval.x; i < interval.y; i++) {   // interval.x 开始点  interval.y 下一个开始点
    int indice = indices[i];                        // 获取这个点在geometry上的index
    int camera_index = indice / (ndepth * farea);   // 第几个camera
    int fm_inner_index = indice % farea;            // 在平面hw中的index
    half depth_weight = depth_weights[indice];      // 该点对应的weight 
    unsigned int camera_feature_offset = (camera_index * farea + fm_inner_index) * nchannel + feature_block;// 获取camera_feature的偏移量，因为camera_feature比geometry 多了80的feature维度
    combined_half feature = *(combined_half*)(camera_feature + camera_feature_offset);     // 获取camera feature

#pragma unroll
    /* 
      将camera feature (80个元素)中，每一个feature都乘以对应的depth_weight。
      回忆一下depth_weight表示的概率分布, 所以乘以depth_weight可以表示这个feature的重要性
      hfma表示的是half-precision floating-point FMA。下面的公式可以理解为是:
      accumulate[j] += feature[j] * depth_weight
      这里面做了累加其实就是为了将一个interval中所有的feature都进行乘以depth_weight之后合并
      #pragma unroll表示将这个for循环展开加速
    */
    for (int j = 0; j < tile_size; j++) {
      accumulate[j] = __hfma(((half*)&feature)[j], depth_weight, accumulate[j]);
    }
  }

#pragma unroll
  // 将累加的结果赋值给bev的feature map
  for (int j = 0; j < tile_size; j++) {
    unsigned int output_offset = interval.z + (feature_block + j) * out_h * out_w;
    output_bevfeat[output_offset] = accumulate[j];
  }
}

class BEVPoolImplement : public BEVPool {
 public:
  virtual ~BEVPoolImplement() {
    if (output_feature_) checkRuntime(cudaFree(output_feature_));
  }

  bool init(const std::vector<int>& camera_shape, unsigned int bev_width, unsigned int bev_height) {
    this->camera_shape_ = camera_shape;
    this->bev_width_ = bev_width;
    this->bev_height_ = bev_height;

    unsigned int C = camera_shape_[1];
    volumn_output_ = C * bev_width * bev_height;
    output_dims_ = {1, (int)C, (int)bev_height, (int)bev_width};
    checkRuntime(cudaMalloc(&output_feature_, volumn_output_ * sizeof(nvtype::half)));
    return true;
  }

  virtual std::vector<int> shape() override { return output_dims_; }

  /*
    BEVPool的forward部分, 负责将camera_feature和depth_weights进行汇总得到的(N, C, D, H, W)的点的信息投影到BEV空间上(N, C, BEVGrid_X_size, BEVGrid_Y_size)
    具体一点就是:
    camera_lidar_features: (N, C, D, H, W) = (6, 80, 118, 32, 88) = 1,993,728 * 80
    BEV_GRID: (1, C, BEVGrid_X_size, BEVGrid_Y_size) = (1, 80, 360, 360) = 129,600 * 80
    这一部分的计算量是相当大，所以采用了一些技巧来进行加速
      - Precomputation
      - Interval Reduction
    Precomputation是体现在camera_geometry的update部分 (详细参考那边的注释)
    Interval Reduction是体现在bevpool_half_pack10_kernel这个核函数里面。
  */

  virtual nvtype::half* forward(const nvtype::half* camera_feature, const nvtype::half* depth_weights,
                                const unsigned int* indices,   // 在geometry中的index
                                const nvtype::Int3* intervals, // 记录着intervals 开始点、结束点、对应bev坐标
                                unsigned int num_intervals,    // intervals分组数量
                                void* stream = nullptr) override {
    unsigned int C, D, H, W;
    C = camera_shape_[1];
    D = camera_shape_[2];
    H = camera_shape_[3];
    W = camera_shape_[4];

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    int thread_x = C / tile_size;  // 8  每10个channel的feature一起处理
    int thread_y = 1024 / thread_x;// 128 一个block内处理128个intervals
    dim3 threads(thread_x, thread_y);
    dim3 blocks(1, int((num_intervals + thread_y - 1) / thread_y)); // 多加一个block，防止取int后少了
    checkRuntime(cudaMemsetAsync(output_feature_, 0x00, volumn_output_ * sizeof(half), _stream));
    
    // 这个kernel是用来做interval reduction的。
    // pack10表示的是一个thread会处理camera_feature_map(80个)中的10个, 同时也是tile size
    // 最终得到的output_feature_其实就是bev的feature map
    checkKernel(
      bevpool_half_pack10_kernel<<<blocks, threads, 0, _stream>>>(
        reinterpret_cast<const half*>(camera_feature), 
        reinterpret_cast<const half*>(depth_weights), C,
        reinterpret_cast<const int3*>(intervals), 
        num_intervals, indices, 
        bev_height_, bev_width_, 
        D, W * H, 
        output_feature_));

    return reinterpret_cast<nvtype::half*>(output_feature_);
  }

 private:
  unsigned int bev_width_ = 0;
  unsigned int bev_height_ = 0;
  std::vector<int> camera_shape_;  // N(num camera), C(feature), D(depth), H(height), W(width)  (6, 80, 118, 32, 88)
  half* output_feature_ = nullptr;
  std::vector<int> output_dims_;
  unsigned int volumn_output_ = 0;
};

// 调用camera命名空间接口下的另外一个实现类: BEVPoolImplement
// 它负责Camera到BEV的投影
std::shared_ptr<BEVPool> create_bevpool(const std::vector<int>& camera_shape, unsigned int bev_width, unsigned int bev_height) {
  std::shared_ptr<BEVPoolImplement> instance(new BEVPoolImplement());
  if (!instance->init(camera_shape, bev_width, bev_height)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion
