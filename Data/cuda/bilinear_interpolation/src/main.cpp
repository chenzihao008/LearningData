#include <opencv2/opencv.hpp>
#include "preprocess.hpp"

cv::Mat procee_gpu(cv::Mat&image,int&tar_h,int&tar_w)
{
cv::Mat h_tar(cv::Size(tar_w,tar_h),CV_8UC3) ;
uint8_t* d_tar = nullptr;
uint8_t* d_src = nullptr;
int src_w = image.cols;
int src_h = image.rows;
int chan = 3;
int target_size = tar_h*tar_w*chan;
int src_size = src_w*src_h*chan;
// cuda maloc
cudaMalloc(&d_src,src_size);
cudaMalloc(&d_tar,target_size);
// cudamemorycp
cudaMemcpy(d_src,image.data,src_size,cudaMemcpyHostToDevice);
// __gloabl__exec
bilinear_gpu(d_tar,d_src, tar_w, tar_h,src_w, src_h);
cudaMemcpy(h_tar.data,d_tar,target_size,cudaMemcpyDeviceToHost);

//cudafree
cudaFree(d_src);
cudaFree(d_tar);

return h_tar;
}

int main(){
cv::Mat image;
image = cv::imread("data/deer.png",1);
int tag_h = 480;
int tag_w = 320;

cv::Mat result = procee_gpu(image,tag_h,tag_w);
cv::imwrite("test.png",result);
return 0;
}