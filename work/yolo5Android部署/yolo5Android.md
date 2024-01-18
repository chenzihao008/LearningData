yolo5Android部署
===
测试环境：windows
参考地址：https://blog.csdn.net/qq_60943902/article/details/132440203
# 注意
1. 需先下载java，并且配置好环境变量
    ```
    变量1
    JAVA_HOME
    %JAVA_HOME17%

    变量2
    CLASSPATH
    .;%JAVA_HOME%\lib\dt.jar;%JAVA_HOME%\lib\tools.jar

    变量3
    JAVA_HOME17
    自己的java17路径
    ```
    配置好后，用该命令测试 java -version
2. ncnn-android-yolov5-master/build.gradle,将红色方框处更换为7.3.0 
3. ncnn-android-yolov5-master/gradle/wrapper,将红色方框处更换为7.4 
   - 第2 第3点配置错误会导致如下错误
        ```
        module java.base does not "opens java.io" to unnamed module @23fb9545
        ```
# 测试结果
- 原图
![](./picture/原图.png)
- 识别结果
![image](./picture/gpu识别结果.png)


