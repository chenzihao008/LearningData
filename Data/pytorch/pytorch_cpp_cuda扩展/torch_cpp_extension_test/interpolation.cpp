#include <torch/extension.h>
#include "utils.hpp"

torch::Tensor trilinearinterpolation(torch::Tensor feats,torch::Tensor point)
{   
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinearinterpolation_cu(feats,point);
}

torch::Tensor trilinearinterpolation_bw(const torch::Tensor dL_dfeats_interp,torch::Tensor feats,torch::Tensor point)
{   
    CHECK_INPUT(dL_dfeats_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(point);
    return trilinearinterpolation_bw_cu(dL_dfeats_interp,feats,point);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m)
{
    m.def("trilinearinterpolation",&trilinearinterpolation);
    m.def("trilinearinterpolation_bw",&trilinearinterpolation_bw);
}