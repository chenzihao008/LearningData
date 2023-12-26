参考地址：https://zhuanlan.zhihu.com/p/601961139
# step1 torch中写好新增的算子
[step1](step1_plugin_torch2onnx.py)
# step2 创建tensorrt plugin
## 大概流程
- 创建logger
- 正常创建builder、network、config、parser
- 解析onnx
    - 创建plugin ，算子运算内容
    - 创建plugincreater，调用plugin算子和获取weight的过程
## 代码
- 两个重要类
    - Plugin类：插件类，插件的具体实现,继承IPluginV2DynamicExt
    - PluginCreator类：插件工厂类，用来根据需求创建插件，调用插件是从这里走，继承IPluginCreator
[plugincpp](./plugin.cpp)
[pluginhpp](./plugin.hpp)
[代码中标识//重要的也是改的比较多的]

# step3 plugin测试

