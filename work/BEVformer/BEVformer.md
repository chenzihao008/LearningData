BEVformer
===
![网络架构](./picture/bevformer.png)
# 论文总结

## 主要贡献
1. 构建稠密BEV表征，加入时序特征
## 消融实验
1. spatial cross-attention不同范围内的对比结果
   ![image](./picture/%E4%B8%8D%E5%90%8C%E8%8C%83%E5%9B%B4%E5%86%85%E4%B8%8Bcrossatten%E5%AF%B9%E6%AF%94.png)
   - 在local范围的内的attention是性能最好的
2. 不同bev grid大小以及layer数量对比
   ![image](./picture/不同bev大小和layer数量对比.png)
   - 在降低bev grid大小以及减少layer数量均对nds和map有较大影响