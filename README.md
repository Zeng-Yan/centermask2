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

| 本仓库脚本                                                   | 功能                               |
| ------------------------------------------------------------ | ---------------------------------- |
| [convert_model_into_onnx.py](https://github.com/Zeng-Yan/centermask2/blob/master/convert_model_into_onnx.py) | 将模型导出为ONNX格式               |
| [tester.py](https://github.com/Zeng-Yan/centermask2/blob/master/tester.py) | ONNX模型的推理                     |
| [visualizer.py](https://github.com/Zeng-Yan/centermask2/blob/master/visualizer.py) | 对结果的可视化                     |
| [preprocess_inputs_to_bin.py](https://github.com/Zeng-Yan/centermask2/blob/master/preprocess_inputs_to_bin.py) | 将数据进行预处理后保存为二进制文件 |
| [postprocess_bin_outputs.py](https://github.com/Zeng-Yan/centermask2/blob/master/postprocess_bin_outputs.py) | 对二进制输出进行后处理及验证精度   |



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

![segmentation](https://github.com/Zeng-Yan/centermask2/blob/master/pics/semantic%20segmentation.png?raw=True)

**2. FPN是干什么的？**

FPN以层级的形式提取图像的特征图。

一个backbone模型提取特征图，如下图左侧，这些特征图被FPN处理得到新特征图，如下图右侧。

![FPN](https://github.com/Zeng-Yan/centermask2/blob/master/pics/FPN.png?raw=True)

**3. FCOS是干什么的？**

FCOS用于目标检测，在图像上框出目标物体。

FCOS基于FPN构建，使用FPN得到不同尺度的特征图，并在各个特征图上分别进行预测。

FCOS的每一个特征图也会对应两个分支，一个分支用于分类，另一个分支用于回归预测框的四个距离。特征图上的每一个点都对应一个Center-ness的值，这个值会与该点所预测的得分相乘，从而降低离目标中心更远的点所预测出的bound-box的得分，从而提升模型的性能。

![FCOS](https://github.com/Zeng-Yan/centermask2/blob/master/pics/FCOS.png?raw=True)

**4. Detectron2是什么？**

Detecron2是一个目标检测框架，centermask是基于Detecron2来开发的。

**5. CenterMask是什么东西？**

有了前面的理解后，就明白centermask在做什么了。centermask是一个语义分割模型，它主要包含三个部分：

(a) backbone 是用于抽取图像特征的网络，在论文中这个backbone是VoVNetV2 

(b) FCOS 用于框定图像中不同的目标

(c) SAG-Mask 用于在FCOS检测出的RoI(Region of Interest)上分割出具体的目标。

![centermask](https://github.com/Zeng-Yan/centermask2/blob/master/pics/centermask.png?raw=T)

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

tracing过程将执行模型的forward函数并记录所执行的计算图，并转为onnx格式。

在此之前您可能需要修改您的forward函数，确保其输入输出均为tensor数据类型，例如detectron2中forward输出是一个Instances对象，您应该将其还原为tensor的形式。另一方面，您应该尽量将预处理和后处理过程从forward函数中剥离出来。

在实现本仓库时，detectron2对导出ONNX的支持并不完善，为了在不修改detectron2代码的情况下实现对模型的修改，这里简单粗暴地从`detectron2.modeling.meta_arch`中删除了原来使用的RCNN，在自己实现修改版本的RCNN后重新加入`detectron2.modeling.meta_arch`中。具体地，修改版RCNN继承了detectron2的RCNN模型简化了其forward过程，仅保留了推理会执行的分支，同时去掉了前后处理操作。

```python
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN


def single_flatten_to_tuple(wrapped_outputs: object) -> tuple:
    """
    模型输出被[Instances.fields]的形式封装，将不同的输出拆解出来组成元组
    :return: (locations, mask_scores, pred_boxes, pred_classes, pred_masks, scores)
    """
    field = wrapped_outputs.get_fields()
    tuple_outputs = (field['locations'], field['mask_scores'],
                     field['pred_boxes'].tensor, field['pred_classes'],
                     field['pred_masks'], field['scores'])
    return tuple_outputs


class GeneralizedRCNN(RCNN):
    def forward(self, img_tensors: torch.Tensor) -> tuple:
        """
        A simplified GeneralizedRCNN for converting pth into onnx,
        without processing (preprocessing and postprocessing) and branches not used in inference
        """
        assert not self.training

        features = self.backbone(img_tensors)
        images = FakeImageList(img_tensors)
        proposals, _ = self.proposal_generator(images, features, None)  # Instance[pred_boxes, scores, pred_classes, locations]
        results, _ = self.roi_heads(images, features, proposals, None)
        results = single_flatten_to_tuple(results[0])
        return results
```

修改完RCNN后，从注册机中删除原来的RCNN并重新注册。

```python
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
```

剥离detectron2的预处理过程，主要是图片的读取，Resize，Normalize及Padding。您应该仔细参考源代码和论文确保您的预处理过程和源模型是一致的。

```python
def get_sample_inputs(path: str) -> list:
    """
    从路径读取一张图片，并按最短边缩放到800，且最长边不超过1333
    :return: [{"image": tensor, "height": tensor, "width": tensor}]
    """
    # load image from path
    original_image = detection_utils.read_image(path, format="BGR")
    height, width = original_image.shape[:2]
    # resize
    aug = T.ResizeShortestEdge([MIN_EDGE_SIZE, MIN_EDGE_SIZE], MAX_EDGE_SIZE)  # [800, 800], 1333
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    dic_img = {"image": image, "height": height, "width": width}
    return [dic_img]


def single_preprocessing(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize and pad the input images.
    """
    # Normalize
    pixel_mean = torch.tensor([103.53, 116.28, 123.675]).view(-1, 1, 1)
    pixel_std = torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1)
    image_tensor = (image_tensor - pixel_mean) / pixel_std

    # Padding
    pad_h = FIXED_EDGE_SIZE - image_tensor.shape[1]
    pad_w = FIXED_EDGE_SIZE - image_tensor.shape[2]

    # padding on right and bottom
    image_tensor = nn.ZeroPad2d(padding=(0, pad_w, 0, pad_h))(image_tensor)

    return image_tensor
```

剥离后处理过程。

```python
def single_wrap_outputs(tuple_outputs: any, height=MAX_EDGE_SIZE, width=MAX_EDGE_SIZE) -> list:
    """
    将元组形式的模型输出重新封装成[Instances.fields]的形式
    """
    instances = Instances((height, width))
    tuple_outputs = [torch.tensor(x)[:50] for x in tuple_outputs]
    instances.set('locations', tuple_outputs[0])
    instances.set('mask_scores', tuple_outputs[1])
    instances.set('pred_boxes', Boxes(tuple_outputs[2]))
    instances.set('pred_classes', tuple_outputs[3])
    instances.set('pred_masks', tuple_outputs[4])
    instances.set('scores', tuple_outputs[5])

    return [instances]


def detector_postprocess(
    results: Instances, h: int, w: int, mask_threshold: float = 0.5
) -> Instances:
    """
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    """
    results = Instances((h, w), **results.get_fields())

    scale = MIN_EDGE_SIZE / min(h, w)
    new_h = int(np.floor(h * scale))
    new_w = int(np.floor(w * scale))
    if max(new_h, new_w) > MAX_EDGE_SIZE:
        scale = MAX_EDGE_SIZE / max(new_h, new_w) * scale

    scale_x, scale_y = 1/scale, 1/scale

    # Rescale pred_boxes
    output_boxes = results.pred_boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    results = results[output_boxes.nonempty()]

    # Rescale pred_masks
    roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
    results.pred_masks = roi_masks.to_bitmasks(
        results.pred_boxes, h, w, mask_threshold
    ).tensor

    return results


def postprocess(instances: list, height=MAX_EDGE_SIZE, width=MAX_EDGE_SIZE) -> list:
    """
    Rescale the output instances to the target size.

    :param instances: list[Instances]
    :param height: int
    :param width: int
    :return: [{"instances": Instances}]
    """
    processed_results = []
    for results_per_image in instances:
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results
```

准备好经过预处理的样例输入，指明输入输出和onnx保存路径，然后您就可以尝试导出onnx了，其核心代码如下。再次强调tracing以样例输入获取计算图，因此您需要指定动态轴说明您的输入输出在哪些维度上是变化的。

```python
# get inputs
if args.pic_file:
    img_path = args.pic_file
else:
    img_path = os.environ['DETECTRON2_DATASETS'] + '/coco/val2017/000000000139.jpg'
batched_inputs = get_sample_inputs(img_path)  # read and resize
inputs = single_preprocessing(batched_inputs[0]['image']).unsqueeze(0)  # preprocessing

# build torch model
model = build_model(cfg)
path_pth = cfg.MODEL.WEIGHTS
# check_keys(model, torch.load(path_pth)['model'])  # compare keys
DetectionCheckpointer(model).load(path_pth)  # load weights
model.eval()

input_names = ['img']
output_names = ['locations', 'mask_scores', 'pred_boxes', 'pred_classes', 'pred_masks', 'scores']
dynamic_axes = {
    'locations': [0],
    'mask_scores': [0],
    'pred_boxes': [0],
    'pred_classes': [0],
    'pred_masks': [0],
    'scores': [0]
}
onnx_path = 'centermask2.onnx'
torch.onnx.export(model, inputs, onnx_path,
                  input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                  opset_version=11, verbose=True)
```

此时，您已经可以通过运行`convert_model_into_onnx.py`文件来得到ONNX了：

```
python convert_model_into_onnx.py --config-file "centermask2/configs/centermask/zy_model_config.yaml"  --version 11 --verbose-on     MODEL.WEIGHTS "centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
```

**3.ONNX调试**

从torch模型转为onnx模型的过程中可能会出现诸多问题，特别是当模型结构复杂时。多数情况下，您需要知晓出现问题的ONNX算子是由哪里的源代码所产生的。

为此，首先确保`torch.onnx.export`中的参数 `verbose=True`，打印导出过程的详细信息。您也可以使用输出重定向或者`nohup`将信息保存到文件中便于后续查看。

如下所示的样例导出信息含义为：输出变量`pred_classes`由输入变量`2640`经过`maskiou_head.py`第`55`行代码中的`Cast`运算所产生。

```
%pred_classes : Long(50, strides=[1], requires_grad=0, device=cpu) = onnx::Cast[to=7](%2640) 
#./centermask2/centermask/modeling/centermask/maskiou_head.py:55:0
```

若您安装了Netron，则您可以在模型中搜索该输出变量找到该Cast算子节点，这样您就可以将ONNX中的算子节点和源代码关联起来了。

一些问题可能出现在ONNX的导出，推理过程中导致报错或者精度损失。例如：

1. 变量维度失配，多数情况下这是由于tracing过程中部分变量被误作为常量记录所导致的，您可以定位到源码，将int变量变为tensor变量，或者增加维度变换操作强制保持维度的一致等等。
2. 由于opset版本导致的数据类型或输入输出不一致，您可以在onnx的GitHub中找到不同opset版本中的算子说明，确认其支持的数据类型和输入输出，然后您可以在源代码中进行变量或类型变换来规避。
3. 您选用的opset版本或者您部署的后端不支持某个算子，那么您需要修改源代码用其他手段，或者修改onnx用其他算子来替代该算子的功能。demos中的两个例子可能会对您有所启发，展示了替代特殊情景下的nonzero算子的样例。
4. 您部署的后端有ONNX算子集所没有的自定义算子，您可以在源代码中通过一个类来伪造这个算子使得ONNX能够正常导出。

```python 
class RoiExtractor(torch.autograd.Function):
    @staticmethod
    def forward(self, f0, f1, f2, rois, aligned=0, finest_scale=56, pooled_height=7, pooled_width=7, pool_mode='avg', roi_scale_factor=0, sample_num=0, spatial_scale=[0.125, 0.0625, 0.03125]):
        """
        feats (torch.Tensor): feats in shape (batch, 256, H, W).
        rois (torch.Tensor): rois in shape (k, 5).
        return:
            roi_feats (torch.Tensor): (k, 256, pooled_width, pooled_width)
        """

        # phony implementation for shape inference
        k = rois.shape[0]
        roi_feats = torch.rand((k, 256, 14, 14)) * 5 - 5
        return roi_feats

    @staticmethod
    def symbolic(g, f0, f1, f2, rois, aligned=0, finest_scale=56, pooled_height=14, pooled_width=14):
        roi_feats = g.op('RoiExtractor', f0, f1, f2, rois, aligned_i=0, finest_scale_i=56, pooled_height_i=pooled_height, pooled_width_i=pooled_width, pool_mode_s='avg', roi_scale_factor_i=0, sample_num_i=0, spatial_scale_f=[0.125, 0.0625, 0.03125], outputs=1)
        return roi_feats


if torch.onnx.is_in_onnx_export():  # 导出onnx时替换自定义算子
    output_size = self.output_size[0]
    pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
    roi_feats = RoiExtractor.apply(x[0], x[1], x[2], pooler_fmt_boxes, 1, 56, output_size, output_size)
    return roi_feats
```



**3. 测试与结果可视化**

TODO

