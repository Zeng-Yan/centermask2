## README

### 目录


> CONTENT:
>
> 1. 简介
> 2. 模型理解
> 3. 环境配置和运行
> 4. 模型推理部署(TODO)

### 简介

本仓库提供对于CenterMask模型（基于Detecron2开发）推理部署的实现，可以作为Pytorch模型导出ONNX格式的示例，以及关于Detecron2开发模型的部署的补充（目前Detecron2只完善了面向Caffe2的部署相关API）。

关于本仓库的文件说明：

根目录下centermask2文件夹为模型代码，其中部分文件在源模型代码上做了修改，以规避模型部署时出现的错误。

（TODO：提供diff文件说明详细的修改情况）

关于将模型导出为ONNX格式，请查看convert_model_into_onnx.py

关于ONNX模型的推理请查看tester.py

关于对结果的可视化请查看visualizer.py

关于将数据进行预处理后保存为二进制文件，请查看preprocess_inputs_to_bin.py

关于对二进制输出进行后处理及验证精度请查看postprocess_bin_outputs.py

### 模型理解

> REFER：
>
> 1. centermask模型论文 [CenterMask : Real-Time Anchor-Free Instance Segmentation]([1911.06667.pdf (arxiv.org)](https://arxiv.org/pdf/1911.06667.pdf))
> 2. FPN模型论文 [Feature Pyramid Networks for Object Detection]([Feature Pyramid Networks for Object Detection (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf))
> 3. FCOS模型论文 [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355.pdf)
> 4. FCOS理解 [目标检测论文解读（一）:FCOS原理解读]([目标检测论文解读（一）:FCOS原理解读 - 简书 (jianshu.com)](https://www.jianshu.com/p/fadaa61133fb))

简要地过一遍centermask论文和源代码，发现一些关键：词语义分割，FPN，FCOS，Detectron2

**1. 什么是语义分割？**

标识出图片中不同类型的对象（通过掩码矩阵来实现）。

![img](https://img-blog.csdn.net/20180310091534193)

**2. FPN是干什么的？**

FPN以层级的形式提取图像的特征图。

一个backbone模型提取特征图，如下图左侧，这些特征图被FPN处理得到新特征图，如下图右侧。

![image-20210701145523832](C:\Users\windf\AppData\Roaming\Typora\typora-user-images\image-20210701145523832.png)

**3. FCOS是干什么的？**

FCOS用于目标检测，在图像上框出目标物体。

FCOS基于FPN构建，使用FPN得到不同尺度的特征图，并在各个特征图上分别进行预测。

FCOS的每一个特征图也会对应两个分支，一个分支用于分类，另一个分支用于回归预测框的四个距离。特征图上的每一个点都对应一个Center-ness的值，这个值会与该点所预测的得分相乘，从而降低离目标中心更远的点所预测出的bound-box的得分，从而提升模型的性能。

![image-20210701153001382](C:\Users\windf\AppData\Roaming\Typora\typora-user-images\image-20210701153001382.png)

**4. Detectron2是什么？**

Detecron2是一个目标检测框架，centermask是基于Detecron2来开发的。

**5. CenterMask是什么东西？**

有了前面的理解后，就明白centermask在做什么了。centermask是一个语义分割模型，它主要包含三个部分：

(a) backbone 是用于抽取图像特征的网络，在论文中这个backbone是VoVNetV2 

(b) FCOS 用于框定图像中不同的目标

(c) SAG-Mask 用于在FCOS检测出的RoI(Region of Interest)上分割出具体的目标。

![image-20210701162029107](C:\Users\windf\AppData\Roaming\Typora\typora-user-images\image-20210701162029107.png)

**6. 进一步了解Detectron2**

> REFER:
>
> 1. [GitHub - facebookresearch/detectron2: Detectron2 is FAIR's next-generation platform for object detection, segmentation and other visual recognition tasks.](https://github.com/facebookresearch/detectron2)
> 2. [Write Models — detectron2 0.5 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/write-models.html)
> 3. [(目录结构)Detectron2入门教程_Wanderer001的博客-CSDN博客_detectron2](https://blog.csdn.net/weixin_36670529/article/details/104021823)

detectron2使用工厂模式

使用detecron2编写新的模型组件，例如一个新的backbone模型，按照以下的方式编写和使用

继承基类，重写方法，并注册

```python
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class ToyBackbone(Backbone):
  def __init__(self, cfg, input_shape):
    super().__init__()
    # create your own backbone
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=16, padding=3)

  def forward(self, image):
    return {"conv1": self.conv1(image)}

  def output_shape(self):
    return {"conv1": ShapeSpec(channels=64, stride=16)}
```

```python
cfg = ...   # read a config
cfg.MODEL.BACKBONE.NAME = 'ToyBackbone'   # or set it in the config file
model = build_model(cfg)  # it will find `ToyBackbone` defined above
```

**7. Centermask网络的具体组成和构建**

模型定义 detectron2/detectron2/modeling/meta_arch/build.py 

参数通过cfg配置

默认参数 /detectron2/config/defaults.py 

默认的模型结构 _C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

Centermask的模型结构使用GeneralizedRCNN

GeneralizedRCNN是在detecron2/modeling/meta_arch/rcnn.py里定义的

GeneralizedRCNN 包含三个主要的部件：backbone，proposal_generator，roi_heads。

```python
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):

```

backbone，proposal_generator，roi_heads三个部件的具体实现为：

```python
print(cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.PROPOSAL_GENERATOR.NAME, cfg.MODEL.ROI_HEADS.NAME)
```

GeneralizedRCNN，FCOS，CenterROIHeads 。

我们要实现的centermask具体版本如下：

| Method          | Backbone | lr sched | inference time | mask AP  | box AP   | download                                                     |
| --------------- | -------- | -------- | -------------- | -------- | -------- | ------------------------------------------------------------ |
| **CenterMask2** | V2-39    | 3x       | **0.050**      | **39.7** | **44.2** | [model](https://dl.dropbox.com/s/tczecsdxt10uai5/centermask2-V-39-eSE-FPN-ms-3x.pth) \| [metrics](https://dl.dropbox.com/s/rhoo6vkvh7rjdf9/centermask2-V-39-eSE-FPN-ms-3x_metrics.json) |

对应的配置文件

configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml

### 环境配置和运行

**1. 依赖配置**

> 从0开始的配置，一般服务器会提供CUDA，CONDA环境，具体信息询问相关负责人

配置CUDA环境变量

```ruby
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.2/lib64
```

安装Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/zeng/miniconda3/bin:$PATH"
```

根据本机的CUDA和CUDNN版本安装GPU版本的Pytorch=1.8，参考官网[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

根据本机CUDA版本和Pytorch版本安装Detectron2，Detectron2更新较快，安装指令参照GitHub

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html 
```

**2. 数据集准备**

> REFFER:
>
> 1. 数据集官网 [COCO - Common Objects in Context (cocodataset.org)](https://cocodataset.org/#download)
> 2. 验证集数据及对应的annotation：val 2017 http://images.cocodataset.org/zips/val2017.zip annotation http://images.cocodataset.org/annotations/annotations_trainval2017.zip
>
> 通常服务器上会准备好数据集，具体信息询问相关负责人。

准备模型需要的coco2017数据集，按官网下载https://cocodataset.org/

然后按照detectron2要求的数据调整文件目录格式，

对于detectron2原生支持的一些数据集，这些数据集被假定存在于一个名为“data/”的目录中，在您启动程序的目录下。它们需要有以下目录结构:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

完成下载后配置数据集所在位置的环境变量

```linux
export DETECTRON2_DATASETS=/home/btree/hzx/centernet2/data
```

**3. 模型权重下载**

下载Backbone模型权重（用于模型训练）：vovnet39_ese_detectron2.pth

```linux
wget -P /root/zeng/centermask2/ https://www.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth?dl=1
```

下载整个模型的权重：centermask2-V-39-eSE-FPN-ms-3x.pth

```linux
wget -P /root/zeng/centermask2/ https://dl.dropboxusercontent.com/s/tczecsdxt10uai5/centermask2-V-39-eSE-FPN-ms-3x.pth
```

**4. 运行模型**

调整配置文件，配置自己的模型权重位置和输出文件夹，建议新建一个自己的配置文件configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml

```yaml
_BASE_: "Base-CenterMask-VoVNet.yaml"
MODEL:
  WEIGHTS: "https://www.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth?dl=1"
  VOVNET:
    CONV_BODY : "V-39-eSE"
SOLVER:
  STEPS: (210000, 250000)
  MAX_ITER: 270000
OUTPUT_DIR: "output/centermask/CenterMask-V-39-ms-3x"
```

修改为configs/centermask/zy_model_config.yaml

```yaml
_BASE_: "Base-CenterMask-VoVNet.yaml"
MODEL:
  WEIGHTS: "/root/zy/centermask2-master/vovnet39_ese_detectron2.pth"
  VOVNET:
    CONV_BODY : "V-39-eSE"
SOLVER:
  STEPS: (210000, 250000)
  MAX_ITER: 270000
OUTPUT_DIR: "output/centermask/zy_model_output"
```

进行模型训练：

```linux
python train_net.py --config-file "configs/centermask/zy_model_config.yaml" --num-gpus 8
```

### 模型推理部署

**1. 环境配置**

> REFFER:
>
> 1. ONNX [GitHub - onnx/onnx: Open standard for machine learning interoperability](https://github.com/onnx/onnx)
> 2. ONNX Runtime [ONNX Runtime (ORT) - onnxruntime](https://onnxruntime.ai/docs/)
> 3. Netron  [GitHub - lutzroeder/netron: Visualizer for neural network, deep learning, and machine learning models](https://github.com/lutzroeder/Netron)

安装onnx

```
conda install -c conda-forge numpy protobuf==3.16.0 libprotobuf=3.16.0
conda install -c conda-forge onnx
```

安装onnxruntime（用于ONNX推理），安装本机情况选择安装指令

```
pip install onnxruntime
```

安装onnxsim（用于优化ONNX，可选）

```
pip install onnx-simplifier
```

一些推荐的工具便于调试：

> 1. PDB [10分钟教程掌握Python调试器pdb - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37294138)
>
> 2. Netron（查看ONNX结构）

**2. 导出ONNX**

TODO

**3. 测试与结果可视化**

TODO


