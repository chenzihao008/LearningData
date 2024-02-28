#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor trilinearinterpolation_cu(torch::Tensor feats,torch::Tensor point);
torch::Tensor trilinearinterpolation_bw_cu(const torch::Tensor dL_dfeats_interp,torch::Tensor feats,torch::Tensor point);