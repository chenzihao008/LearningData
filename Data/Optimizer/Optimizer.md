Optimizer
---
- 动量（Momentum）解决方向问题;
- AdaGrad（自适应梯度）和RMSProp（均方差传播）解决步长问题;
- Adam 结合Momentum和RMSProp;

# BGD&SGD
- BGD：最原始的梯度下降算法，为了计算在整个训练集的loss上的梯度，需要使用训练集全部数据
- SGD（随机梯度下降）：只使用一个mini batch的数据来计算，相当于用minibatch的loss上的梯度去近似计算整个训练集的loss梯度
公式如下：
$$
\theta_{i+1}=\theta_{i}+\eta\nabla(\theta)
$$

# Momentum

# Adagrad

# RMSProp

# Adam

# AdamW