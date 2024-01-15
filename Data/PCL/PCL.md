PCL-基础点云处理算法
===
官网：https://pointclouds.org/
# 安装
参考地址：https://blog.csdn.net/weixin_41836738/article/details/121451965
- 注意先安装依赖再安装pcl
    - vtk需要先源码安装
- 可能会报错 in /usr/lib/x86_64-linux-gnu may be hidden by files in: xxx.conda
    - https://zhuanlan.zhihu.com/p/95497832
    - 把PATH中conda的路径去掉，再重新cmake和make
# 基础理解
参考地址：https://github.com/MNewBie/PCL-Notes
# 测试
## fileter
参考地址：https://zhuanlan.zhihu.com/p/377926459
1. 去除异常点
2. 降采样
3. 
## feature
## keypoints
## registration
## kdtree
## octree
## segmentation
## sample_consensus
## surface
## recognition
