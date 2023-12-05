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
# 三、nvidia docker 
## 1.NVIDIA 获取已有docker容器地址 
- 该地址是查看有版本container：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags
- 改地址查看不同版本container包含什么版本的cuda、tensorrt： https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html

## 2.启动镜像
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


# 五、 vscoed配置
## 1. vscode插件安装
![image](../picture/tensorrt/vs插件.png)
## 2. 配置
0.  agt-get install bear
1. make
```
make前记得修改../../config/Makefile.config文件内的cuda配置
	查看bear版本dpkg -l |grep bear
	2.4 版本 使用： bear make -j16
	3.0 以上版本使用：bear -- make -j16
```


2. 按ctl+shift+p 选择 c_cpp_properties.json，会在.vscode 生成c_cpp_properties.json文件
```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++14",
            "intelliSenseMode": "linux-gcc-x64",
            "compileCommands": "${workspaceFolder}/compile_commands.json" #新增，compile_commands.json 是bear make后才有的
        }
    ],
    "version": 4
}
```

3. 配置language
查看地址：https://code.visualstudio.com/docs/languages/identifiers#:~:text=Language%20Identifiers%20In%20Visual%20Studio%20Code%2C%20each%20language,to%20a%20language%3A%20%22files.associations%22%3A%20%7B%20%22%2A.myphp%22%3A%20%22php%22%20%7D
```
在.vscode中创建settings.json 
加入
{
    "files.associations": {
        "*.cu": "cuda-cpp"
    }
}
```
4. task 配置
```
按ctl+shift+p 
选择 Configure.Task   
选择create   
选择other
会在.vscode文件夹中生成tasks.json 文件
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "make",  #修改
            "type": "shell",
            "command": "make -j16" #修改
        }
    ]
}
```
5. debug 配置
```
按ctl+shift+p 
选择dubug：Add  Configure
选择CUDAC++(CUDA-GDB)
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "${workspaceFolder}/trt-cuda"  #修改  trt-cuda 要与config/Makefile.config 内APP的名字一致
        },
        {
            "name": "CUDA C++: Attach",
            "type": "cuda-gdb",
            "request": "attach"
        }
    ]
}

bebug还是会报错error while loading shared libraries: libncursesw.so.5: cannot open shared object file
原因说明:https://blog.csdn.net/winter99/article/details/117464598
使用命令（首先确定有/lib/x86_64-linux-gnu/libncursesw.so.6）：sudo ln -s /lib/x86_64-linux-gnu/libncursesw.so.6 /lib/x86_64-linux-gnu/libncursesw.so.5
```