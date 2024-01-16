#include "cuda_runtime_api.h"
#include "utils.hpp"
#include "preprocess.hpp"

__global__ void  bilinear_kernel(uint8_t* d_tar,uint8_t* d_src,int tar_w, int tar_h,int src_w,int src_h,float scale_h,float scale_w)
{
int x = blockIdx.x*blockDim.x+threadIdx.x;
int y = blockIdx.y*blockDim.y+threadIdx.y;

// lefttop  right bottle
int src_x1 = floor(x*scale_w);
int src_y1 = floor(y*scale_h);
int src_x2 = floor(x*scale_w)+1;
int src_y2 = floor(y*scale_h)+1;
// compute w and h 
float tw = x*scale_w-src_x1;
float th = y*scale_h-src_y1;

float lt_area = th*tw;
float ld_area = (1-th)*tw;
float rt_area = th*(1-tw);
float rd_area = (1-th)*(1-tw);

// src id 
int lt_id = (src_y1*src_w+src_x1)*3;
int ld_id = (src_y2*src_w+src_x1)*3;
int rt_id = (src_y1*src_w+src_x2)*3;
int rd_id = (src_y2*src_w+src_x2)*3;
// tar id 
int tar_id =  (y*tar_w+x)*3;

// r
d_tar[tar_id] = d_src[lt_id+0]*rd_area+d_src[ld_id+0]*rt_area+d_src[rt_id+0]*ld_area+d_src[rd_id+0]*lt_area;

// g
d_tar[tar_id+1] = d_src[lt_id+1]*rd_area+d_src[ld_id+1]*rt_area+d_src[rt_id+1]*ld_area+d_src[rd_id+1]*lt_area;

// b
d_tar[tar_id+2] = d_src[lt_id+2]*rd_area+d_src[ld_id+2]*rt_area+d_src[rt_id+2]*ld_area+d_src[rd_id+2]*lt_area;

}

void bilinear_gpu(uint8_t* d_tar,uint8_t* d_src,
                            int tar_w,int tar_h,
                            int src_w,int src_h)
{
dim3 dimBlock(16,16,1);
dim3 dimGrid(tar_w/16+1,tar_h/16+1,1);
// scale size
float scale_h = (float)src_h/tar_h;
float scale_w = (float)src_w/tar_w;

bilinear_kernel <<<dimGrid, dimBlock>>> (d_tar,d_src,tar_w,tar_h,src_w,src_h,scale_h,scale_w);

}

void jiust_xdim_bilinear_gpu(uint8_t* d_tar,uint8_t* d_src,
                            int tar_w,int tar_h,
                            int src_w,int src_h)
{
dim3 dimBlock(16,16,1);
dim3 dimGrid(tar_w/16+1,tar_h/16+1,1);
// scale size
float scale_h = (float)src_h/tar_h;
float scale_w = (float)src_w/tar_w;

bilinear_kernel <<<dimGrid, dimBlock>>> (d_tar,d_src,tar_w,tar_h,src_w,src_h,scale_h,scale_w);

}