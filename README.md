# 开源项目整理&实验
## 一、单目&多目&多传感器3D目标检测
### [MONOCON3D](https://github.com/chenzihao008/monocon_rp)
  - 年份：2022
### [DERT3D](work/DERT3D/DERT3D.md)
  - 年份：2021
  - sparse 的方式，主要表现在稀疏的query查询
  - 输入：images(环视图)
### [BEVformer](work/BEVformer/BEVformer.md)
  - 年份：2022
  - dense 的方式，主要表现在dense的bevfeature上
  - 输入: images(环视图)
### [FUTR3D](work/FUTR3D/FUTR3D.md)
  - 年份：2023
  - lidar+radar+camera融合
### [RCBEV](work/RCBEV/RCBEV.md)
## 二、BEV Occupancy
- 关注点：
    1. miou、模型性能
    2. occ精细程度
    3. 感知距离
    4. 训练数据获取
        - 点云稠密程度决定了gtocc的稠密程度
### [SelfOcc]
  - 年份：2023
  - 自监督的方式生成occ
### [SURROUNDOCC](work/surroundocc/surroundocc.md)
  - 年份：2023
  - lidar数据生成gt,只有image输入
  - 通过多帧lidar合并以及泊松表面重建，解决occ稀疏问题

### [FlashOCC](work/FlashOCC/FlashOCC.md)
  - 年份：2023
  - 只有image输入，全程2d conv 
  - 缓解3D CONV 带来的计算开销
### [UniOcc](work/UniOcc/UniOcc.md)
  - 年份：2023
  - 只有image输入，增加2d的depth和sam监督
  - 通过体渲染增加Depth和Semantic render的监督，缓解对于occ标注的依赖
## 三、[CUDA-BEVFUSION：image和lidar融合](work/CUDA-BEVFusion/CUDA-BEVFUSION.md)
## 四、2D目标检测&跟踪
### [YOLO8]()
- [TRT Deployment](work/yolov8/yolov8_depolyment.md)
### [RTDETR]()
### [fast RCNN]()


### [ByteTrack]()
## 五、NERF
### [GaussianEditor：可编辑nerf](work/NERF/GaussianEditor/GaussianEditor.md)
### [Repaint123]()
### [DVGO]()
### [NERF](work/NERF/NERF/NERF.md)
### [NERF++: 将前景和背景分开]()
### [Mip-NeRF360：解决漂浮物问题](work/NERF/Mip-NeRF360/Mip-NeRF360.md)


## 八、[mmdetection3D部署相关内容](work/mmdetection3D_deployment/mmdetection3D_deploy.md)
# 资料整理
（空缺内容在本地文件上待整理上来）
## 1.[感知算法框架](Data/感知算法框架.md)
## 2.[Transformer/Swin-transformer/deformable](Data/Transformer.md)
## 3.[Loss](Data/loss/loss.md)
## 4.[Optimizer](Date/)
## 5.[Normalization]()
## 6.[激活函数]()
## 7.[多卡训练]()
## 8.[模型优化知识](Data/模型优化知识.md)
## 9.[Cuda&Tensorrt&量化](Data/Cuda&Tensorrt.md)
## 11.训练trick汇总
## 12.C++笔记
## 13.[数据集](Data/dataset.md) 
## 14.open3d
## 15.mmdetection3D
## 17.[openCV]()
## 16.[卡尔曼滤波]()
## 17.[基础PCL](Data/PCL/PCL.md)
## 18.[毫米波雷达](Data/毫米波雷达/RADAR.md)