# 资料整理
## 1.[感知算法框架](Data/感知算法框架.md)
## 2.[Transformer/Swin-transformer/deformable](Data/Transformer.md)
## 3.[Loss](Data/loss.md)
## 4.Optimize
## 5.Normalization
## 6.激活函数
## 7.多卡训练
## 8.[模型优化知识](Data/模型优化知识.md)
## 9.[Cuda&Tensorrt&量化](Data/Cuda&Tensorrt.md)
## 11.训练trick汇总
## 12.C++笔记
## 13.[数据集](Data/dataset.md) 
## 14.open3d
## 15.mmdetection3D


# 开源项目整理&实验
## 一、2D目标检测&跟踪
### YOLO8
## [Deployment](work/yolov8/yolov8_depolyment.md)
### RTDETR
### ByteTrack

## 二、单目3D目标检测
### [MONOCON3D](https://github.com/chenzihao008/monocon_rp)
- 年份：2022

## 三、BEV Occupancy
### [SelfOcc：自监督的方式生成occ]
- 年份：2023
### [SURROUNDOCC: lidar数据生成gt,只有image输入](work/surroundocc/surroundocc.md)
- 年份：2023
- 解决输出occ稀疏问题
### [FlashOCC: 只有image输入，全程2d conv ](work/FlashOCC/FlashOCC.md)
- 年份：2023
- 缓解3D CONV 带来的计算开销
### [UniOcc: 只有image输入，增加2d的depth和sam监督](work/UniOcc/UniOcc.md)
- 年份：2023
- 通过体渲染增加depth和segmentation的监督，缓解对于occ标注的依赖
### [Occ3D]TODO
### [OctreeOcc](work/OctreeOcc/OctreeOcc.md)
### [BEVDet]
## 四、[CUDA-BEVFUSION：image和lidar融合](work/CUDA-BEVFusion/CUDA-BEVFUSION.md)
## 五、NERF
TODO
### [Repaint123]
### [DVGO]
### [NERF](work/NERF/NERF/NERF.md)
### [Mip-NeRF360](work/NERF/Mip-NeRF360/Mip-NeRF360.md)


## 六、[mmdetection3D部署相关内容](work/mmdetection3D_deployment/mmdetection3D_deploy.md)
