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

#include "common/check.hpp"
#include "common/launch.cuh"
#include "lidar-voxelization.hpp"

namespace bevfusion {
namespace lidar {

static __device__ inline uint64_t hash(uint64_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;
}

static __device__ inline void insert_to_hash_table(const uint32_t key, uint32_t *value, const uint32_t hash_size,
                                                   uint32_t *hash_table) {
  uint64_t hash_value = hash(key);// 获取hash值
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;//获取 实际点云数量*2 的余数 作为偏移量 
  uint32_t empty_key = UINT32_MAX;
  while (true) {
    uint32_t pre_key = atomicCAS(hash_table + slot, empty_key, key); //((hash_table + slot) == compare ? key : hash_table + slot)
    if (pre_key == empty_key) {
      hash_table[slot + hash_size / 2 /*offset*/] = atomicAdd(value, 1); // 在hashtable中对应位置添加atomicAdd(value, 1) 表示第几个voxel
      break;
    } else if (pre_key == key) {
      break;
    }
    slot = (slot + 1) % (hash_size / 2);
  }
}

static __device__ inline uint32_t lookup_hash_table(const uint32_t key, const uint32_t hash_size, const uint32_t *hash_table) {
  uint64_t hash_value = hash(key);
  uint32_t slot = hash_value % (hash_size / 2) /*key, value*/;
  uint32_t empty_key = UINT32_MAX;
  int cnt = 0;
  while (true /* need to be adjusted according to data*/) {
    cnt++;
    if (hash_table[slot] == key) {
      return hash_table[slot + hash_size / 2];
    } else if (hash_table[slot] == empty_key) {
      return empty_key;
    } else {
      slot = (slot + 1) % (hash_size / 2);
    }
  }
  return empty_key;
}

static __global__ void build_hash_table_kernel(size_t points_size, const half *points, VoxelizationParameter param,
                                               unsigned int *hash_table, unsigned int *real_voxel_num) {
  int point_idx = cuda_linear_index;
  if (point_idx >= points_size) return;
  // 获取点云的xyz值
   float px = points[param.num_feature * point_idx];
  float py = points[param.num_feature * point_idx + 1];
  float pz = points[param.num_feature * point_idx + 2];
  // 转换为voxel的xyz index
  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);
  if (voxel_idx < 0 || voxel_idx >= param.grid_size.x) return;

  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  if (voxel_idy < 0 || voxel_idy >= param.grid_size.y) return;

  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if (voxel_idz < 0 || voxel_idz >= param.grid_size.z) return;
  // 计算voxel的一维 index
  unsigned int voxel_offset = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx;
  insert_to_hash_table(voxel_offset, real_voxel_num, points_size * 2 * 2, hash_table);// points_size 当前帧点云数量
}

template <CoordinateOrder order>
static __device__ void save_result_by_order(uint4 *output, uint x, uint y, uint z);

template <>
__device__ void save_result_by_order<CoordinateOrder::XYZ>(uint4 *output, uint x, uint y, uint z) {
  *output = make_uint4(0, x, y, z);
}

template <>
__device__ void save_result_by_order<CoordinateOrder::ZYX>(uint4 *output, uint x, uint y, uint z) {
  *output = make_uint4(0, z, y, x);
}

template <CoordinateOrder order>
static __global__ void voxelization_kernel(size_t points_size, const half *points, VoxelizationParameter param,
                                           unsigned int *hash_table, unsigned int *num_points_per_voxel, float *voxels_temp,
                                           unsigned int *voxel_indices) {
  int point_idx = cuda_linear_index;
  if (point_idx >= points_size) return;

  float px = points[param.num_feature * point_idx];// 点云坐标x，单位m
  float py = points[param.num_feature * point_idx + 1];// 点云坐标y，单位m
  float pz = points[param.num_feature * point_idx + 2];// 点云坐标z，单位m

  if (px < param.min_range.x || px >= param.max_range.x || py < param.min_range.y || py >= param.max_range.y ||
      pz < param.min_range.z || pz >= param.max_range.z) {
    return;
  }
  // 转换为voxel index xyz
  int voxel_idx = floorf((px - param.min_range.x) / param.voxel_size.x);
  int voxel_idy = floorf((py - param.min_range.y) / param.voxel_size.y);
  int voxel_idz = floorf((pz - param.min_range.z) / param.voxel_size.z);
  if ((voxel_idx < 0 || voxel_idx >= param.grid_size.x)) {
    return;
  }
  if ((voxel_idy < 0 || voxel_idy >= param.grid_size.y)) {
    return;
  }
  if ((voxel_idz < 0 || voxel_idz >= param.grid_size.z)) {
    return;
  }

  unsigned int voxel_offset = (voxel_idz * param.grid_size.y + voxel_idy) * param.grid_size.x + voxel_idx; // 改点对应voxel在一维上的index

  // scatter to voxels
  // 在voxel_offset key对应hash地址中记录该voxel收纳了多少个point，超过10个话直接返回
  unsigned int voxel_id = lookup_hash_table(voxel_offset, points_size * 2 * 2, hash_table);
  if (voxel_id >= param.max_voxels) {
    return;
  }
  // 在featuremap的voxels_temp 插入该voxel下的点
  unsigned int current_num = atomicAdd(num_points_per_voxel + voxel_id, 1);
  if (current_num < param.max_points_per_voxel) {
    unsigned int dst_offset = voxel_id * (param.num_feature * param.max_points_per_voxel) + current_num * param.num_feature;
    unsigned int src_offset = point_idx * param.num_feature;
    for (int feature_idx = 0; feature_idx < param.num_feature; ++feature_idx) {
      voxels_temp[dst_offset + feature_idx] = points[src_offset + feature_idx];
    }

    // now only deal with batch_size = 1
    // since not sure what the input format will be if batch size > 1
    save_result_by_order<order>(&((uint4 *)voxel_indices)[voxel_id], voxel_idx, voxel_idy, voxel_idz);
  }
}

static __global__ void reduce_mean_kernel(size_t num_voxels, float *voxels_temp, unsigned int *num_points_per_voxel,
                                          int max_points_per_voxel, int feature_num, half *voxel_features) {
  int voxel_idx = cuda_linear_index;
  if (voxel_idx >= num_voxels) return;

  num_points_per_voxel[voxel_idx] =
      num_points_per_voxel[voxel_idx] > max_points_per_voxel ? max_points_per_voxel : num_points_per_voxel[voxel_idx];
  int valid_points_num = num_points_per_voxel[voxel_idx];
  int offset = voxel_idx * max_points_per_voxel * feature_num;
  // 各voxel第一个点的xyz替换成该voxel下所有有效点的平均值
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    for (int point_idx = 0; point_idx < valid_points_num - 1; ++point_idx) {
      voxels_temp[offset + feature_idx] += voxels_temp[offset + (point_idx + 1) * feature_num + feature_idx];
    }
    voxels_temp[offset + feature_idx] /= valid_points_num;
  }

  // move to be continuous 只获取均值转换成fp16，连续储存
  for (int feature_idx = 0; feature_idx < feature_num; ++feature_idx) {
    int dst_offset = voxel_idx * feature_num;
    int src_offset = voxel_idx * feature_num * max_points_per_voxel;
    voxel_features[dst_offset + feature_idx] = __float2half(voxels_temp[src_offset + feature_idx]);
  }
}

nvtype::Int3 VoxelizationParameter::compute_grid_size(const nvtype::Float3 &max_range, const nvtype::Float3 &min_range,
                                                      const nvtype::Float3 &voxel_size) {
  nvtype::Int3 size;
  size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
  size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
  size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));
  return size;
}

class VoxelizationImplement : public Voxelization {
 public:
  virtual ~VoxelizationImplement() {
    if (hash_table_) checkRuntime(cudaFree(hash_table_));
    if (voxels_temp_) checkRuntime(cudaFree(voxels_temp_));

    if (d_voxel_features_) checkRuntime(cudaFree(d_voxel_features_));
    if (d_voxel_num_) checkRuntime(cudaFree(d_voxel_num_));
    if (d_voxel_indices_) checkRuntime(cudaFree(d_voxel_indices_));

    if (d_real_num_voxels_) checkRuntime(cudaFree(d_real_num_voxels_));
    if (h_real_num_voxels_) checkRuntime(cudaFreeHost(h_real_num_voxels_));
  }

  //分配device memory,这里可以参考一下CUDA-CenterPoint
  bool init(VoxelizationParameter param) {
    this->param_ = param;
    this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1}; //输出grid xyz尺寸

    this->hash_table_size_ = param_.max_points * 2 * 2 * sizeof(unsigned int);//hash_table_size_=300000*2*2*4
    this->voxels_temp_size_ = param_.max_voxels * param_.max_points_per_voxel * param_.num_feature * sizeof(float);// voxels_temp_size_= 160000*10*5*4
    this->voxel_features_size_ = param_.max_voxels * param_.max_points_per_voxel * param_.num_feature * sizeof(half);// 160000*10*5*2
    this->voxel_num_size_ = param_.max_voxels * sizeof(unsigned int);//160000*4
    this->voxel_idxs_size_ = param_.max_voxels * 4 * sizeof(unsigned int);//160000*4*4

    checkRuntime(cudaMalloc(&hash_table_, hash_table_size_));
    checkRuntime(cudaMalloc(&voxels_temp_, voxels_temp_size_));
    checkRuntime(cudaMalloc(&d_voxel_features_, voxel_features_size_));
    checkRuntime(cudaMalloc(&d_voxel_num_, voxel_num_size_));
    checkRuntime(cudaMalloc(&d_voxel_indices_, voxel_idxs_size_));
    checkRuntime(cudaMalloc(&d_real_num_voxels_, sizeof(unsigned int)));
    checkRuntime(cudaMallocHost(&h_real_num_voxels_, sizeof(unsigned int)));
    return true;
  }

  // points and voxels must be of half type
  // 这里面的实现可以参考一下CUDA-CenterPoint
  virtual void forward(const nvtype::half *points, int num_points, void *stream, CoordinateOrder output_order) override {
    cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
    const half *_points = reinterpret_cast<const half *>(points);
    checkRuntime(cudaMemsetAsync(hash_table_, 0xff, hash_table_size_, _stream));
    checkRuntime(cudaMemsetAsync(voxels_temp_, 0xff, voxels_temp_size_, _stream));
    checkRuntime(cudaMemsetAsync(d_voxel_num_, 0, voxel_num_size_, _stream));
    checkRuntime(cudaMemsetAsync(d_real_num_voxels_, 0, sizeof(unsigned int), _stream));
    // 通过核函数构建hashtabel，因为同一个voxel可能会存在多个点云
    cuda_linear_launch(build_hash_table_kernel, //核函数
                        _stream,                
                        num_points,             //当前帧lidarpoints数量，对应kernel启动数量
                         _points,               //点云数量
                         param_,                //参数
                         hash_table_,           //hashtabel地址
                         d_real_num_voxels_);   //记录实际使用voxel数量，初始化为0
    checkRuntime(cudaMemcpyAsync(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost, _stream));

    // for difference output order  将对应的feature选择前10个插入到featuremap中
    if (output_order == CoordinateOrder::XYZ) {
      cuda_linear_launch(voxelization_kernel<CoordinateOrder::XYZ>, _stream, num_points, _points, param_, hash_table_,
                         d_voxel_num_, voxels_temp_, d_voxel_indices_);
      this->output_grid_size_ = {(int)param_.grid_size.x, (int)param_.grid_size.y, (int)param_.grid_size.z + 1};
    } else if (output_order == CoordinateOrder::ZYX) {
      cuda_linear_launch(voxelization_kernel<CoordinateOrder::ZYX>, _stream, num_points, _points, param_, hash_table_,
                         d_voxel_num_, voxels_temp_, d_voxel_indices_);
      this->output_grid_size_ = {(int)param_.grid_size.z + 1, (int)param_.grid_size.y, (int)param_.grid_size.x};
    } else
      Assertf(false, "Invalid output_order: %d", static_cast<int>(output_order));

    checkRuntime(cudaStreamSynchronize(_stream));
    // 取单个voxel内所有有效点的平均值
    real_num_voxels_ = *h_real_num_voxels_;
    cuda_linear_launch(reduce_mean_kernel, _stream, real_num_voxels_, voxels_temp_, d_voxel_num_, param_.max_points_per_voxel,
                       param_.num_feature, d_voxel_features_);
  }

  virtual unsigned int num_voxels() override { return real_num_voxels_; }

  virtual unsigned int voxel_dim() override { return param_.num_feature; }

  virtual unsigned int indices_dim() override { return 4; }

  virtual std::vector<int> grid_size() override { return output_grid_size_; }

  virtual const void *indices() override { return d_voxel_indices_; }

  virtual const void *features() override { return d_voxel_features_; }

  virtual CoordinateOrder order() override { return order_; }

 private:
  CoordinateOrder order_ = CoordinateOrder::NoneOrder;
  VoxelizationParameter param_;
  unsigned int real_num_voxels_ = 0;
  std::vector<int> output_grid_size_;

  unsigned int *hash_table_ = nullptr;
  float *voxels_temp_ = nullptr;
  unsigned int *d_real_num_voxels_ = nullptr;
  unsigned int *h_real_num_voxels_ = nullptr;
  unsigned int *d_voxel_num_ = nullptr;
  half *d_voxel_features_ = nullptr;
  unsigned int *d_voxel_indices_ = nullptr;
  unsigned int hash_table_size_;
  unsigned int voxels_temp_size_;
  unsigned int voxel_features_size_;
  unsigned int voxel_idxs_size_;
  unsigned int voxel_num_size_;
};

// 调用lidar命名空间接口下的一个实现类: VoxelizationImplement
// 主要用来负责将点云进行体素化
std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter param) {
  std::shared_ptr<VoxelizationImplement> impl(new VoxelizationImplement());
  if (!impl->init(param)) {
    impl.reset();
  }
  return impl;
}

};  // namespace lidar
};  // namespace bevfusion
