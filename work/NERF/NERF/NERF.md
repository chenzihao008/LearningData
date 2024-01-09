NERF
===
# 相关知识
1. SDF
2. volume rendre（体渲染）
    1. rays生成
        - rays数据结构:5D(x,y,z,θ,φ)位置+观察方向
        - rays生成策略
        - rays数量
    2.  体渲染公式
        - 密度
        - 颜色吸收能力
        - 剩余光强

# 复现
- 环境
    - ubuntu 20.04
    - python=3.8
    - cuda=11.3
    - torch=1.11.0 
    - torchvision=0.12.0
    - 4090 24G
- 耗时：3hours
![image](./picture/nerf_training.png)
- 结果

