BEVFUSION
![image](./BEVFUSION%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6.png)
- BEVFusion：https://github.com/mit-han-lab/bevfusion
- https://github.com/mit-han-lab 这里有很多很好好的代码，值得花时间看
===
# 模块分析
1. Camera Encoder 
    - 可以选择Swin，也可以选择Resnet
2. LiDAR Encoder
    - 选用的是CenterPoint中的SCN(Sparse Convolution Network)
3. Camera-to-BEV
    - 预测每个像素的深度分布，之后使用BEVPool转换到BEV空间
4. LiDAR-to-BEV
    - 提取特征后在Z方向进行flatten，转换到BEV空间
5. BEV Encoder
    - 将BEV空间下的Camera feature和LiDAR feature进行融合concat，再通过几个conv来关联起来
6. 输出
    - 3D detection(x, y, z, w, h, vx, vy, sin, cos, score)
    - Segmentation(CUDA-BEVFusion中暂时没有实现)