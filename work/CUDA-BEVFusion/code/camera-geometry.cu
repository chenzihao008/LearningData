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
#include <thrust/sort.h>

#include "camera-geometry.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensor.hpp"

namespace bevfusion {
namespace camera {

struct GeometryParameterExtra : public GeometryParameter {
  unsigned int D;
  nvtype::Float3 dx;
  nvtype::Float3 bx;
  nvtype::Int3 nx;
};

static __forceinline__ __device__ float dot(const float4& T, const float3& p) { return T.x * p.x + T.y * p.y + T.z * p.z; }

static __forceinline__ __device__ float project(const float4& T, const float3& p) {
  return T.x * p.x + T.y * p.y + T.z * p.z + T.w;
}

static __forceinline__ __device__ float3 inverse_project(const float4* T, const float3& p) {
  float3 r;
  r.x = p.x - T[0].w;
  r.y = p.y - T[1].w;
  r.z = p.z - T[2].w;
  return make_float3(dot(T[0], r), dot(T[1], r), dot(T[2], r));
}

static __global__ void arange_kernel(unsigned int num, int32_t* p) {
  int idx = cuda_linear_index;
  if (idx < num) {
    p[idx] = idx;
  }
}

static __global__ void interval_starts_kernel(unsigned int num, unsigned int remain, unsigned int total, const int32_t* ranks,
                                              const int32_t* indices, int32_t* interval_starts, int32_t* interval_starts_size) {
  int idx = cuda_linear_index;
  if (idx >= num) return;

  unsigned int i = remain + 1 + idx;// remain 是ranks为0部分的数量，也就是不映射到bevgrid内的
  // 我们从这里可以看到，相同的rank的点的interval是一样的
  if (ranks[i] != ranks[i - 1]) {
    unsigned int offset = atomicAdd(interval_starts_size, 1);
    interval_starts[offset] = idx + 1;
  }
}

/*
  intervals[i]是由三个int(x, y, z)组成:  
  - x表示的是这个interval开始的点
  - y表示的是这个interval结束的点
  - z表示的是这个interval在BEVGrid上所对应的grid的id
  在BEVPool的时候，对于每一个interval的计算，直接通过这个索引就好了
*/
static __global__ void collect_starts_kernel(unsigned int num, unsigned int remain, unsigned int numel_geometry,
                                             const int32_t* indices, const int32_t* interval_starts, const int32_t* geometry,
                                             int3* intervals) {
  int i = cuda_linear_index;
  if (i >= num) return;

  int3 val;
  val.x = interval_starts[i] + remain;                                                          // intervals 在 geometry上开始的index
  val.y = i < num - 1 ? interval_starts[i + 1] + remain : numel_geometry - interval_starts[i];  // intervals 在 geometry上结束的index
  val.z = geometry[indices[interval_starts[i] + remain]];                                       // BEVGrid中的坐标
  intervals[i] = val;
}

static void __host__ matrix_inverse_4x4(const float* m, float* inv) {
  double det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
  double invdet = 1.0 / det;
  inv[0] = (m[5] * m[10] - m[9] * m[6]) * invdet;
  inv[1] = (m[2] * m[9] - m[1] * m[10]) * invdet;
  inv[2] = (m[1] * m[6] - m[2] * m[5]) * invdet;
  inv[3] = m[3];
  inv[4] = (m[6] * m[8] - m[4] * m[10]) * invdet;
  inv[5] = (m[0] * m[10] - m[2] * m[8]) * invdet;
  inv[6] = (m[4] * m[2] - m[0] * m[6]) * invdet;
  inv[7] = m[7];
  inv[8] = (m[4] * m[9] - m[8] * m[5]) * invdet;
  inv[9] = (m[8] * m[1] - m[0] * m[9]) * invdet;
  inv[10] = (m[0] * m[5] - m[4] * m[1]) * invdet;
  inv[11] = m[11];
  inv[12] = m[12];
  inv[13] = m[13];
  inv[14] = m[14];
  inv[15] = m[15];
}

/*
  对frustum中的每一个数据赋值(x, y, z), 其中:
  x: 原图大小中的x坐标
  y: 原图大小中的y坐标
  z: 以0.5为刻度，1~60m的距离
*/
static __global__ void create_frustum_kernel(unsigned int feat_width, unsigned int feat_height, unsigned int D,
                                             unsigned int image_width, unsigned int image_height, float w_interval,
                                             float h_interval, nvtype::Float3 dbound, float3* frustum) {
  int ix = cuda_2d_x;
  int iy = cuda_2d_y;
  int id = blockIdx.z;
  if (ix >= feat_width || iy >= feat_height) return;

  unsigned int offset = (id * feat_height + iy) * feat_width + ix;
  frustum[offset] = make_float3(ix * w_interval, iy * h_interval, dbound.x + id * dbound.z);
}

static __global__ void compute_geometry_kernel(unsigned int numel_frustum, const float3* frustum, const float4* camera2lidar,
                                               const float4* camera_intrins_inv, const float4* img_aug_matrix_inv,
                                               nvtype::Float3 bx, nvtype::Float3 dx, nvtype::Int3 nx, unsigned int* keep_count,
                                               int* ranks, nvtype::Int3 geometry_dim, unsigned int num_camera,
                                               int* geometry_out) {
  int tid = cuda_linear_index;
  if (tid >= numel_frustum) return;

  // 每一个线程负责frustum中的一个点。frustum中每一个元素有三个float(x, y, depth)
  float3 point = frustum[tid];

  // 在每一个相机中寻找这个点在BEV坐标系下的点
  for (int icamerea = 0; icamerea < num_camera; ++icamerea) {

    // 畸变的map
    float3 projed = inverse_project(img_aug_matrix_inv, point);
    projed.x *= projed.z;
    projed.y *= projed.z;

    // img->camera的map
    projed = make_float3(
      dot(camera_intrins_inv[4 * icamerea + 0], projed), 
      dot(camera_intrins_inv[4 * icamerea + 1], projed),
      dot(camera_intrins_inv[4 * icamerea + 2], projed));

    // camera->lidar的map
    projed = make_float3(
      project(camera2lidar[4 * icamerea + 0], projed), 
      project(camera2lidar[4 * icamerea + 1], projed),
      project(camera2lidar[4 * icamerea + 2], projed));

    int _pid = icamerea * numel_frustum + tid;
    int3 coords;

    // 这样我们就可以找到frustum的这个点，应该投影在BEVGrid上的点(x, y, z)了
    coords.x = int((projed.x - (bx.x - dx.x / 2.0)) / dx.x);
    coords.y = int((projed.y - (bx.y - dx.y / 2.0)) / dx.y);
    coords.z = int((projed.z - (bx.z - dx.z / 2.0)) / dx.z);

    // 这样我们可以把这个点在geometry中的offset与BEVGrid的坐标对应上了, 当我们再寻找img->BEVGrid的时候，直接可以通过这个geometry_out来找
    geometry_out[_pid] = (coords.z * geometry_dim.z * geometry_dim.y + coords.x) * geometry_dim.x + coords.y;

    bool kept = coords.x >= 0 && coords.y >= 0 && coords.z >= 0 && coords.x < nx.x && coords.y < nx.y && coords.z < nx.z;
    if (!kept) {
      ranks[_pid] = 0;
    } else {
      // 这个keep_count就是用来保留最终有多少个img上的点可以投影到BEVGrid上去的
      atomicAdd(keep_count, 1);
      ranks[_pid] = (coords.x * nx.y + coords.y) * nx.z + coords.z; //有些点的rank可能会一样，rank一样的点会在同一个interval中  和 geometry_out[_pid]的值是一样的，因为bevgrid下coords.z=0 nx.z=1
    }
  }
}

class GeometryImplement : public Geometry {
 public:
  virtual ~GeometryImplement() {
    if (counter_host_) checkRuntime(cudaFreeHost(counter_host_));
    if (keep_count_) checkRuntime(cudaFree(keep_count_));
    if (frustum_) checkRuntime(cudaFree(frustum_));
    if (geometry_) checkRuntime(cudaFree(geometry_));
    if (ranks_) checkRuntime(cudaFree(ranks_));
    if (indices_) checkRuntime(cudaFree(indices_));
    if (interval_starts_) checkRuntime(cudaFree(interval_starts_));
    if (interval_starts_size_) checkRuntime(cudaFree(interval_starts_size_));
    if (intervals_) checkRuntime(cudaFree(intervals_));
    if (camera2lidar_) checkRuntime(cudaFree(camera2lidar_));
    if (camera_intrinsics_inverse_) checkRuntime(cudaFree(camera_intrinsics_inverse_));
    if (img_aug_matrix_inverse_) checkRuntime(cudaFree(img_aug_matrix_inverse_));
    if (camera_intrinsics_inverse_host_) checkRuntime(cudaFreeHost(camera_intrinsics_inverse_host_));
    if (img_aug_matrix_inverse_host_) checkRuntime(cudaFreeHost(img_aug_matrix_inverse_host_));
  }

  // 初始化camera->BEV所需要做的Precomputation的信息，重点是intervals, 以及interval_starts
  bool init(GeometryParameter param) {
    static_cast<GeometryParameter&>(param_) = param;

    param_.D = (unsigned int)std::round((param_.dbound.y - param_.dbound.x) / param_.dbound.z);

    param_.bx = nvtype::Float3(
      param_.xbound.x + param_.xbound.z / 2.0f, // -54+0.3/2
      param_.ybound.x + param_.ybound.z / 2.0f, // -54+0.3/2
      param_.zbound.x + param_.zbound.z / 2.0f);// -10+20/2=0

    param_.dx = nvtype::Float3(
      param_.xbound.z, //0.3
      param_.ybound.z, //0.3
      param_.zbound.z);//20
    //bev grid的大小
    param_.nx = nvtype::Int3(
      static_cast<int>(std::round((param_.xbound.y - param_.xbound.x) / param_.xbound.z)),//360
      static_cast<int>(std::round((param_.ybound.y - param_.ybound.x) / param_.ybound.z)),//360
      static_cast<int>(std::round((param_.zbound.y - param_.zbound.x) / param_.zbound.z)));//1

    cudaStream_t stream = nullptr;
    float w_interval = (param_.image_width - 1.0f) / (param_.feat_width - 1.0f); //8
    float h_interval = (param_.image_height - 1.0f) / (param_.feat_height - 1.0f); //8

    // 这里的frustrum可以理解为视锥, 对于feature_map的大小(88, 32)中的每一个点，我们都有D个(118个)深度估计的值
    // 把每一个相机的frustrum组合在一起就是geometry中的点的个数，200万左右个点, 也就是论文中所说的camera-lidar feature map
    numel_frustum_ = param_.feat_width * param_.feat_height * param_.D; // 88 * 32 * 118
    numel_geometry_ = numel_frustum_ * param_.num_camera; // 88 * 32 * 118 * 6

    checkRuntime(cudaMallocHost(&counter_host_, sizeof(int32_t)));
    checkRuntime(cudaMalloc(&keep_count_, sizeof(int32_t)));                 // 用来表示最终有多少个点可以投影到BEVFrid上
    checkRuntime(cudaMalloc(&frustum_, numel_frustum_ * sizeof(float3)));    // 视锥，大小为 88 * 32 * 118
    checkRuntime(cudaMalloc(&geometry_, numel_geometry_ * sizeof(int32_t))); // 6个视锥, 大小为 6 * 88 * 32 * 118
    checkRuntime(cudaMalloc(&ranks_, numel_geometry_ * sizeof(int32_t)));    // geometry中每一个点都有一个rank, 
    checkRuntime(cudaMalloc(&indices_, numel_geometry_ * sizeof(int32_t)));  // 用来每一个点的index
    checkRuntime(cudaMalloc(&interval_starts_, numel_geometry_ * sizeof(int32_t))); // intervals的数组中，每一个interval都有一个起始点，用来在BEVPool的时候寻址
    checkRuntime(cudaMalloc(&interval_starts_size_, sizeof(int32_t)));       // 用来保存一共有多少个interval
    checkRuntime(cudaMalloc(&intervals_, numel_geometry_ * sizeof(int3)));   // intervals的数组，一个interval会包含多个点

    bytes_of_matrix_ = param_.num_camera * 4 * 4 * sizeof(float);
    checkRuntime(cudaMalloc(&camera2lidar_, bytes_of_matrix_));
    checkRuntime(cudaMalloc(&camera_intrinsics_inverse_, bytes_of_matrix_));
    checkRuntime(cudaMalloc(&img_aug_matrix_inverse_, bytes_of_matrix_));
    checkRuntime(cudaMallocHost(&camera_intrinsics_inverse_host_, bytes_of_matrix_));
    checkRuntime(cudaMallocHost(&img_aug_matrix_inverse_host_, bytes_of_matrix_));

    // 这个kernel是用来创建frustum(视锥), frustum中一共有 88 * 32 * 118个数据， 每一个数据都是float3
    // 分别代表的是原图大小中的x坐标，y坐标，以及以0.5m为刻度的深度
    cuda_2d_launch(
      create_frustum_kernel, 
      stream, 
      param_.feat_width, param_.feat_height, param_.D, 
      param_.image_width, param_.image_height, 
      w_interval, h_interval, 
      param_.dbound, 
      frustum_);

    return true;
  }

  // You can call this function if you need to update the matrix
  // All matrix pointers must be on the host
  /*
    camera geometry的update很重要，主要是负责BEVPool的Precomputation, 从而防止在BEVPool中做大量的计算
    换句话说，就是如果camera位置固定不变，那么BEVPool中各个grid的点所对应的camera的坐标也是固定的，只要预先知道这个关系就好办了
  */
  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* img_aug_matrix,
                      void* stream = nullptr) override {
    Asserts(frustum_ != nullptr,
            "If the excess memory has been freed, then the update call will not be logical for the "
            "program.");

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    // 将各个camera的3D->2D的参数与camera畸变的参数取出
    for (unsigned int icamera = 0; icamera < param_.num_camera; ++icamera) {
      unsigned int offset = icamera * 4 * 4;
      matrix_inverse_4x4(camera_intrinsics + offset, camera_intrinsics_inverse_host_ + offset);
      matrix_inverse_4x4(img_aug_matrix + offset, img_aug_matrix_inverse_host_ + offset);
    }

    // For users, please ensure that the pointer lifecycle is available for asynchronous copying.
    checkRuntime(cudaMemcpyAsync(camera2lidar_, camera2lidar, bytes_of_matrix_, cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(camera_intrinsics_inverse_, camera_intrinsics_inverse_host_, bytes_of_matrix_,
                                 cudaMemcpyHostToDevice, _stream));
    checkRuntime(cudaMemcpyAsync(img_aug_matrix_inverse_, img_aug_matrix_inverse_host_, bytes_of_matrix_, cudaMemcpyHostToDevice,
                                 _stream));
    checkRuntime(cudaMemsetAsync(keep_count_, 0, sizeof(unsigned int), _stream));

    // 这个kernel负责计算frustum中的每一个点所对应的BEVGrid中的坐标, 并跟存储到geometry_中
    // 同时为每一个点都设置一个rank, 相同的rank的点会在同一个interval中
    cuda_linear_launch(
      compute_geometry_kernel, 
      _stream, 
      numel_frustum_, frustum_, 
      reinterpret_cast<const float4*>(camera2lidar_),
      reinterpret_cast<const float4*>(camera_intrinsics_inverse_),
      reinterpret_cast<const float4*>(img_aug_matrix_inverse_), 
      param_.bx, param_.dx, param_.nx, 
      keep_count_,
      ranks_, 
      param_.geometry_dim, param_.num_camera, 
      geometry_);

    checkRuntime(cudaMemcpyAsync(counter_host_, keep_count_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));

    // 这个kernel用来更新indices的信息, 可以暂时理解为geometry中每一个点的index
    // 从0开始计数给indices_为[0,1,2,...]
    cuda_linear_launch(
      arange_kernel, 
      _stream, 
      numel_geometry_, indices_);

    // 使用CUDA中的thrust库中的sort来对ranks_里面的所有数据进行排序, 同时ranks中每一个点所对应的indices_也会根据这个排序重新arrange
    // 根据key进行排序也就是这里的ranks，即从0开始排，例如[0，0，0，1，1，1，2，2]，0是没有投影到bevgrid上geometry_的点;这样做的原因是后续会根据这个ranks的index分组处理相同index的数据
    // - ranks 记录着对应bev的index
    // - indices 记录着自身geometry_的顺序，也就是geometry_ 自身的index
    thrust::stable_sort_by_key(thrust::cuda::par.on(_stream), 
                                ranks_,                         //key的开始
                                ranks_ + numel_geometry_,       //key的last
                                indices_,                       //value
                                thrust::less<int>());           //使用升序 ；降序的话使用thrust::greater<int>()
    checkRuntime(cudaStreamSynchronize(_stream));

    unsigned int remain_ranks = numel_geometry_ - *counter_host_; // 身下没有投影的点的数量
    unsigned int threads = *counter_host_ - 1;                    //表示能够投影在BEVGrid上的点
    checkRuntime(cudaMemsetAsync(interval_starts_size_, 0, sizeof(int32_t), _stream));//分组数量

    // set interval_starts_[0] to 0
    checkRuntime(cudaMemsetAsync(interval_starts_, 0, sizeof(int32_t), _stream));

    // 这个kernel负责寻找每一块inteval的开始的位置，以及interval的总数
    cuda_linear_launch(
      interval_starts_kernel, 
      _stream, 
      threads, remain_ranks, 
      numel_geometry_, ranks_, indices_,
      interval_starts_ + 1, // 记录interval开始的index
      interval_starts_size_);// 记录有多少个interval组，也就是bevgrid 多少个有效点

    checkRuntime(cudaMemcpyAsync(counter_host_, interval_starts_size_, sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
    checkRuntime(cudaStreamSynchronize(_stream));

    // interval_starts_[0] = 0,  and counter += 1
    // 这里的counter_host_已经是interval_starts的个数了
    n_intervals_ = *counter_host_ + 1;

    // 对于所有的interval进行一下排序
    // 从小到达 对interval的各组开始的index 进行排序,例如：原数据[1,*,*,*,*,5,*,*...]->变成[0，0，0，1，5，....]
    thrust::stable_sort(thrust::cuda::par.on(_stream), interval_starts_, interval_starts_ + n_intervals_, thrust::less<int>());

    // 这个kernel负责寻找开始的interval, 并给存储到intervals_里面，供BEVPool的时候使用
    cuda_linear_launch(
      collect_starts_kernel, 
      _stream, 
      n_intervals_, remain_ranks, 
      numel_geometry_, 
      indices_, 
      interval_starts_,
      geometry_, intervals_);
  }

  virtual void free_excess_memory() override {
    if (counter_host_) {
      checkRuntime(cudaFreeHost(counter_host_));
      counter_host_ = nullptr;
    }
    if (keep_count_) {
      checkRuntime(cudaFree(keep_count_));
      keep_count_ = nullptr;
    }
    if (frustum_) {
      checkRuntime(cudaFree(frustum_));
      frustum_ = nullptr;
    }
    if (geometry_) {
      checkRuntime(cudaFree(geometry_));
      geometry_ = nullptr;
    }
    if (ranks_) {
      checkRuntime(cudaFree(ranks_));
      ranks_ = nullptr;
    }
    if (interval_starts_) {
      checkRuntime(cudaFree(interval_starts_));
      interval_starts_ = nullptr;
    }
    if (interval_starts_size_) {
      checkRuntime(cudaFree(interval_starts_size_));
      interval_starts_size_ = nullptr;
    }
    if (camera2lidar_) {
      checkRuntime(cudaFree(camera2lidar_));
      camera2lidar_ = nullptr;
    }
    if (camera_intrinsics_inverse_) {
      checkRuntime(cudaFree(camera_intrinsics_inverse_));
      camera_intrinsics_inverse_ = nullptr;
    }
    if (img_aug_matrix_inverse_) {
      checkRuntime(cudaFree(img_aug_matrix_inverse_));
      img_aug_matrix_inverse_ = nullptr;
    }
    if (camera_intrinsics_inverse_host_) {
      checkRuntime(cudaFreeHost(camera_intrinsics_inverse_host_));
      camera_intrinsics_inverse_host_ = nullptr;
    }
    if (img_aug_matrix_inverse_host_) {
      checkRuntime(cudaFreeHost(img_aug_matrix_inverse_host_));
      img_aug_matrix_inverse_host_ = nullptr;
    }
  }

  virtual unsigned int num_intervals() override { return n_intervals_; }

  virtual unsigned int num_indices() override { return numel_geometry_; }

  virtual nvtype::Int3* intervals() override { return reinterpret_cast<nvtype::Int3*>(intervals_); }

  virtual unsigned int* indices() override { return reinterpret_cast<unsigned int*>(indices_); }

 private:
  size_t bytes_of_matrix_ = 0;
  float* camera2lidar_ = nullptr;
  float* camera_intrinsics_inverse_ = nullptr;
  float* img_aug_matrix_inverse_ = nullptr;
  float* camera_intrinsics_inverse_host_ = nullptr;
  float* img_aug_matrix_inverse_host_ = nullptr;

  float3* frustum_ = nullptr; //视锥？
  unsigned int numel_frustum_ = 0;

  unsigned int n_intervals_ = 0;
  unsigned int numel_geometry_ = 0;
  int32_t* geometry_ = nullptr;
  int32_t* ranks_ = nullptr;
  int32_t* indices_ = nullptr;
  int3*    intervals_ = nullptr;
  int32_t* interval_starts_ = nullptr;
  int32_t* interval_starts_size_ = nullptr;
  unsigned int* keep_count_ = nullptr;
  unsigned int* counter_host_ = nullptr;
  GeometryParameterExtra param_;
};

// 调用camera命名空间接口下的一个实现类: GeometryImplement
std::shared_ptr<Geometry> create_geometry(GeometryParameter param) {
  std::shared_ptr<GeometryImplement> instance(new GeometryImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace camera
};  // namespace bevfusion
