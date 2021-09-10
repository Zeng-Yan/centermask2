# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import math
import sys
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat
from detectron2.modeling.poolers import (
        convert_boxes_to_pooler_format, assign_boxes_to_levels
)


def nonzero(**kwargs) -> torch.tensor:
    """
    由于华为方不支持nozero算子，该函数用topk规避nonzero
    """
    if not torch.onnx.is_in_onnx_export():  # 在线推理时正常执行nonzero算子
        idx = torch.nonzero(kwargs['input'], )
    else:  # 导出onnx时执行的分支
        x = kwargs['input'].to(torch.float32)  # bool/? -> int64 统一数据类型避免奇怪的错误
        k = torch.sum(x)  # 设置k值
        if x.ndim == 1:
            k = torch.min(torch.tensor(x.shape[0]).to(torch.float32), k)  # 一维情况下避免索引越界，op 11 要求min为float
            k = k.reshape(1).to(torch.int64)  # topk的k必须是1d int64
            _, idx = x.topk(k.item())
            idx = idx.unsqueeze(-1)  # [M, 1] 改变形状对应nonzero的输出
        else:  # 输入为二维情况下的执行分支
            fixed_dim = torch.tensor(x.shape[1], dtype=torch.int64)  # [80] 记录固定的列数，以便还原二维索引
            x = x.flatten()  # [N, 80] -> [N*80]  奇怪的地方，这里被onnx换成了reshape
            nms_top = kwargs['nms_top']  # nms_top仅在二维情况下生效
            k = torch.min(nms_top.to(torch.float32), k)  # op 11 要求min为float
            k = k.reshape(1).to(torch.int64)  # topk的k必须是1d int64
            _, col_idx = x.topk(k.item())  # 将二维tensor展平后用topk规避
            col_idx = col_idx.to(torch.int64)  # 增加cast便于onnx修改算子
            row_idx = (col_idx / fixed_dim).floor().to(torch.int64)  # topk在原二维tensor对应的行索引
            # col_idx = col_idx.fmod(fixed_dim).to(torch.int64)  # topk在原二维tensor对应的列索引
            col_idx = (col_idx - row_idx * fixed_dim).to(torch.int64)  # opset9 不支持fmod
            idx = torch.stack((row_idx, col_idx), dim=-1)  # [k, 2] 合并为[[行索引, 列索引], ...]的形式
        idx = idx[0:k[0]]  # 一个无意义的操作来保留onnx中k的计算

    return idx.to(torch.int64)


class RoiExtractor(torch.autograd.Function):
    @staticmethod
    def forward(self, f0, f1, f2, rois, aligned=0, finest_scale=56, pooled_height=7, pooled_width=7,
                         pool_mode='avg', roi_scale_factor=0, sample_num=0, spatial_scale=[0.125, 0.0625, 0.03125]):
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
        # TODO: support tensor list type for feats
        roi_feats = g.op('RoiExtractor', f0, f1, f2, rois, aligned_i=0, finest_scale_i=56, pooled_height_i=pooled_height, pooled_width_i=pooled_width,
                         pool_mode_s='avg', roi_scale_factor_i=0, sample_num_i=0, spatial_scale_f=[0.125, 0.0625, 0.03125], outputs=1)
        return roi_feats


def _img_area(instance):

    device = instance.pred_classes.device
    image_size = instance.image_size
    area = torch.as_tensor(image_size[0] * image_size[1], dtype=torch.float, device=device)
    tmp = torch.zeros((len(instance.pred_classes), 1), dtype=torch.float, device=device)

    return (area + tmp).squeeze(1)


def assign_boxes_to_levels_by_ratio(instances, min_level, max_level, is_train=False):
    """
    Map each box in `instances` to a feature map level index by adaptive ROI mapping function 
    in CenterMask paper and return the assignment
    vector.

    Args:
        instances (list[Instances]): the per-image instances to train/predict masks.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    if is_train:
        box_lists = [x.proposal_boxes for x in instances]
    else:
        box_lists = [x.pred_boxes for x in instances]
    box_areas = cat([boxes.area() for boxes in box_lists])
    img_areas = cat([_img_area(instance_i) for instance_i in instances])

    # Eqn.(2) in the CenterMask paper
    if torch.onnx.is_in_onnx_export():  # 导出onnx时裁剪形状
        img_areas = img_areas[:box_areas.shape[0]]
        box_areas = box_areas[:img_areas.shape[0]]
    level_assignments = torch.ceil(
        max_level - torch.log2(img_areas / box_areas + eps)
    )

    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def assign_boxes_to_levels(box_lists, min_level, max_level, canonical_box_size, canonical_level):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (box_tensor.shape[0], 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
        assign_crit="area",
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            # When there is only one feature map, canonical_level is redundant and we should not
            # require it to be a sensible value. Therefore we skip this assertion
            assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        self.assign_crit = assign_crit #ywlee

    def forward(self, x, instances, is_train=False):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
            is_train (True/False)

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        if is_train:
            box_lists = [x.proposal_boxes for x in instances]
        else:
            box_lists = [x.pred_boxes for x in instances]

        if torch.onnx.is_in_onnx_export():  # 导出onnx时替换自定义算子
            output_size = self.output_size[0]
            pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
            # print(f'\nbefore slice: {pooler_fmt_boxes.shape}')
            # pooler_fmt_boxes = pooler_fmt_boxes[:, 1::]
            # print(f'after slice: {pooler_fmt_boxes.shape}')

            roi_feats = RoiExtractor.apply(x[0], x[1], x[2], pooler_fmt_boxes, 1, 56, output_size, output_size)
            return roi_feats

        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        print(f'\n pooler_fmt_boxes pooler_fmt_boxes[:, 1::]')
        print(f'{pooler_fmt_boxes.shape} {pooler_fmt_boxes[:, 1::].shape}')
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        if self.assign_crit == "ratio":
            level_assignments = assign_boxes_to_levels_by_ratio(
                instances, self.min_level, self.max_level, is_train
            )
        else: #default
            level_assignments = assign_boxes_to_levels(
                box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
            )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = nonzero(input=level_assignments == level).squeeze(1)  # max [50, 1]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)

        return output
