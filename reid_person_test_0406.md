# 从git复现reid - person 部分

_本文件为在 Intel NUC1 上部署 REID_Person 的实际操作步骤。以下步骤均参照 README 文件实行，可作为原 README 文件的补充。_

## 1. 下载权重：  
[百度云] https://pan.baidu.com/s/1ZI_UVUsPC9NKPFmjyF4MiQ  
提取码：1234  

## 2. 上传至 deep_sort/deep/  
    _注意 `.rar` 文件不能被 `unzip` 解压，所以我将其在本地解压后重新压缩为 `.zip` 文件再执行上传操作。_

## 3. 解压
```sh
unzip deep_sort/deep/checkpoint.zip
```

## 4. 新建虚拟环境
```sh
conda create -n reid0406 python=3.7
# 提示是否继续后输入 y
conda activate reid0406
```

## 5. 安装 requirements
```sh
pip install -r requirements.txt # 注意README中的文件名少了s
```
刚说着这里可能会出问题，这问题就来了。报错：
```sh
ERROR: Could not find a version that satisfies the requirement Cython (from versions: none)
ERROR: No matching distribution found for Cython
```
查阅资料发现可能是镜像源的问题。报错的是 Cython 包，尝试手动设置清华镜像源单独安装。
```sh
pip install cython -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装时提示没有安装 `matplotlib >= 2.1.0`，但是 Cython 安装成功了。这里因为 requirements 中含有 matplotlib，我直接再次尝试安装 requirements。
```sh
pip install -r requirements.txt
```
这次是另一个包，同样的问题：
```sh
ERROR: Could not find a version that satisfies the requirement pillow (from versions: none)
ERROR: No matching distribution found for pillow
```
尝试同样的解决办法：
```sh
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```
成功后再安装 requirements：
```sh
pip install -r requirements.txt
```
这次是 `socket.timeout`，可能是暂时的网络问题，我过一段时间再试一次。  
今天再次尝试安装过程成功，但是有几个不安的方面，第一，这次仍然有提示前置依赖未安装，信息如下：
```sh
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
prettytable 2.1.0 requires wcwidth, which is not installed.
gdown 3.13.0 requires filelock, which is not installed.
```
第二，安装的内容中含有 `nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cublas-cu11` 等有 `cuda` 相关字段的包，不能确定所安装的 `torch` 是否能在只有 cpu 的设备上正常运行。  
先尝试继续之后的步骤。

## 6. 上传测试用视频
在 hws 文件夹中搜索含有 input 字段的文件：
```sh
find ~/hws/ -name '*input*'
```
没有 `.mp4` 类型的结果。同样地，在当前文件夹中搜索发现也不存在 `.mp4` 类型的文件。
```sh
find . -name '*.mp4'
```
所以简单选择在buaa项目中提供的 `ori.mp4` 作为输入测试。这里直接选择复制：
```sh
cp /home/ubuntu/yrc/buaaDemo/data/samples/ori.mp4 ./test.mp4
```

## 7. 执行 `person_count.py`
```sh
python3 person_count.py --video_path './test.mp4'
```
报错没有 onnxruntime 包：
```sh
ModuleNotFoundError: No module named 'onnxruntime'
```
直接尝试使用 pip 安装：
```sh
pip install onnxruntime
```
下载到一半又是 http 报错，这次直接重试。  
成功后再次运行。这次缺 easydict，同样尝试 pip 安装。
```sh
 pip install easydict
```
成功后再次运行，这次目测是代码问题：
```sh
AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```
由于报错位置在 torch 包中，所以推测为常见错误，上网搜索解决办法。根据 [https://blog.csdn.net/weixin_43401024/article/details/124428432] 提供的解决办法，进入报错文件直接删除该项缺失的返回值，具体步骤如下：  
打开对应文件：
```sh
nano /home/ubuntu/.conda/envs/reid0406/lib/python3.7/site-packages/torch/nn/modules/upsampling.py
```
使用 `CTRL+W` 后在搜索栏输入 recompute_scale_factor 搜索。之后重复 8 次 `ALT+W` 跳转至最后一条搜索结果，在该位置输入 # 将该行变为注释，然后在上一行末尾删去多余的 ',' 并补充缺失的 ')'。`CTRL+X`后输入 Y 保存后退出。  
```python
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
                             # recompute_scale_factor=self.recompute_scale_factor)
```
再次运行成功。

## 8. 查看当前结果
由于命令行没有输出保存位置，所以在刚刚运行的文件中查找。  
发现了几处疑惑的地方，第一是输入参数中的 device 默认值为 cuda:0，但是刚刚没有 cuda 报错。第二是输出似乎是设置了 display 参数后才会有保存输出结果，但是似乎限定了 `.jpg` 格式。而且默认值为 `TRUE`，但是运行中及结束后没有 temp_img 子目录。  
调了挺久现在终于能存结果了，不知道原本保存输出的步骤是在哪里实现的。以下是修改位置，全部在 `person_count.py` 中。  
在 `class yolo_reid`, `def deep_sort()`函数中，于 115 行后添加如下几行：
```python
        temp_path, vid_writer = None, None
        fourcc='mp4v'
        save_path = './output.mp4'
```
然后在同函数的 206 行后添加如下几行：
```python
            if temp_path != save_path:  # new video
                temp_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
    
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            vid_writer.write(ori_img)
```
此时已经可以将结果保存在 `./output.mp4` 中。在再次测试以前，先上传示例所用待测视频 input_reid.mp4 后再次运行。本次结果中识别出了人的位置但是不能很好地区分不同的人且没能将一度离开镜头范围的对象正确识别为同一个人。

## 9. 其他修改
在 README 中还提到了一项针对 x86 设备运行本工程的改动，需要修改 `reid_clothes.py` 文件，使用如下命令在当前目录中搜索发现不存在以该名称命名的文件。
```sh
find . -name 'reid_clothes.py'
```
从名称推断，这项修改可能应当在服装识别工程中使用。
