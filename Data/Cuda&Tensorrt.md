# Tensorrt环境配置
根据tensorrt版本再选择相应的cuda和cudnn

# 一、tensorrt安装	
- 可以查看Tensorrt匹配的cuda和cudnn版本地址：https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html
- Tensorrt 保存在任意文件夹都可以的,记得在 ~/.bashrc 上加上PATH即可
```
export PATH="/home/ubuntu/Public/TensorRT-8.5.1.7/bin:$PATH"
export LD_LIBRARY_PATH="/home/ubuntu/Public/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH"
```

# 二、cudnn安装教程
- 官方地址：https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
- 使用 Tar File Installation

# 三、NVIDIA 获取已有docker容器地址 
- 该地址是查看有版本container：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags
- 改地址查看不同版本container包含什么版本的cuda、tensorrt： https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html

# 四、启动镜像
```
#!/bin/sh/

docker run -it \                            #直接进入容器内
        --name trt_starter_${1} \           #指定容器的名字
  --gpus all \  #可以使用所有gpu
        -v /tmp/.X11-unix:/tmp/.X11-unix \ # 可以远程进行gui操作
        -v /home/ubuntu/Code:/home/ubuntu/Code\ # 挂载文件
  -p 8090:22 \                                # 主体的8090对container的22
        -e DISPLAY=:1 \
        trt_starter:cuda11.4-cudnn8-tensorrt8.2_${1}  #对应镜像版本
```


# 五、vscode插件安装
![image](../picture/tensorrt/vs插件.png)
