#include <torch/extension.h>
#include "utils.hpp"

template<typename scalar_t>
__global__ void trilinearinterpolation_kernel(
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> feat_interp,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> feats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> points,
        int N,int F)
{
    /*out dim return */
    int f = blockIdx.x*blockDim.x+threadIdx.x;
    int n = blockIdx.y*blockDim.y+threadIdx.y;
    if (n>=N || f>=F){
        return;
    }
    // 三线性插值
    // pints [-1,1]
    const scalar_t u = (points[n][0]+1)/2; //转换到0-1
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;

    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[n][f] = (1-u)*( a*feats[n][0][f]+
                                b*feats[n][1][f]+
                                c*feats[n][2][f]+
                                d*feats[n][3][f])+
                            u*( a*feats[n][4][f]+
                                b*feats[n][5][f]+
                                c*feats[n][6][f]+
                                d*feats[n][7][f]);
        
    return ;
}

torch::Tensor trilinearinterpolation_cu(torch::Tensor feats,torch::Tensor point)
{
    const int N = feats.size(0),F = feats.size(2);
    torch::Tensor feat_interp = torch::zeros({N,F},feats.options());
    dim3 dimblock(16,16);
    dim3 dimgrid((F-1)/16+1,(N-1)/16+1);
    /*
    AT_DISPATCH_FLOATING_TYPES 动态分发函数根据feats.type()的类型，确定实例化哪种函数

    */ 
    AT_DISPATCH_FLOATING_TYPES(  
        feats.type(),
        "trilinearinterpolation_cu",
        ([&]{
        trilinearinterpolation_kernel<scalar_t><<<dimgrid,dimblock>>>(
            // packed_accessor 数据格式转换，因为torch：：tensor不能直接传入kernel
            // 数字2、3 代表维度数量
            // torch::RestrictPtrTraits 代表跟其他输入没有交集
            // size_t 表示index 为long unsigned int
            feat_interp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            feats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            point.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            N,F
            );

        })
    );

    return feat_interp;
}


template<typename scalar_t>
__global__ void trilinearinterpolation_bw_kernel(
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dL_dfeats_interp,
        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dL_dfeats,
        // torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> feats,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> points,
        int N,int F)
{
    /*out dim return */
    int f = blockIdx.x*blockDim.x+threadIdx.x;
    int n = blockIdx.y*blockDim.y+threadIdx.y;
    if (n>=N || f>=F){
        return;
    }
    // 三线性插值
    // pints [-1,1]
    const scalar_t u = (points[n][0]+1)/2; //转换到0-1
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;

    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    dL_dfeats[n][0][f] = dL_dfeats_interp[n][f]*(1-u)*a;
    dL_dfeats[n][1][f] = dL_dfeats_interp[n][f]*(1-u)*b;
    dL_dfeats[n][2][f] = dL_dfeats_interp[n][f]*(1-u)*c;
    dL_dfeats[n][3][f] = dL_dfeats_interp[n][f]*(1-u)*d;
    dL_dfeats[n][4][f] = dL_dfeats_interp[n][f]*(u)*a;
    dL_dfeats[n][5][f] = dL_dfeats_interp[n][f]*(u)*b;
    dL_dfeats[n][6][f] = dL_dfeats_interp[n][f]*(u)*c;
    dL_dfeats[n][7][f] = dL_dfeats_interp[n][f]*(u)*d;
        
    return ;
}

torch::Tensor trilinearinterpolation_bw_cu(
    const torch::Tensor dL_dfeats_interp,
    torch::Tensor feats,
    torch::Tensor point
    )
{
    const int N = feats.size(0),F = feats.size(2);
    torch::Tensor dL_dfeats = torch::zeros({N,8,F},feats.options());
    dim3 dimblock(16,16);
    dim3 dimgrid((F-1)/16+1,(N-1)/16+1);
    /*
    AT_DISPATCH_FLOATING_TYPES 动态分发函数根据feats.type()的类型，确定实例化哪种函数

    */ 
    AT_DISPATCH_FLOATING_TYPES(  
        feats.type(),
        "trilinearinterpolation_bw_cu",
        ([&]{
        trilinearinterpolation_bw_kernel<scalar_t><<<dimgrid,dimblock>>>(
            // packed_accessor 数据格式转换，因为torch：：tensor不能直接传入kernel
            // 数字2、3 代表维度数量
            // torch::RestrictPtrTraits 代表跟其他输入没有交集
            // size_t 表示index 为long unsigned int
            dL_dfeats_interp.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            dL_dfeats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            // feats.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            point.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            N,F
            );

        })
    );

    return dL_dfeats;
}