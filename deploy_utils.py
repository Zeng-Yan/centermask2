# defining misc funcs using in deploying
# Author: zengyan
# Final: 21.08.28

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
sys.path.append('./centermask2')

import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes, ROIMasks
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import add_export_config
from centermask.config import get_cfg

MIN_EDGE_SIZE = 800
MAX_EDGE_SIZE = 1333
FIXED_EDGE_SIZE = 1344


def to_numpy(tensor: torch.Tensor) -> np.array:
    """
    convert tensor to array
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def check_keys(model: nn.Module, state_dict: dict) -> None:
    """
    验证模型的各种权重key和保存的state_dict是否匹配
    """
    keys_model = set(model.state_dict().keys())
    keys_state = set(state_dict.keys())
    keys_miss = keys_model - keys_state
    keys_used = keys_model & keys_state
    print(f'{len(keys_model)} keys of model, {len(keys_state)} keys of state_dict')
    print(f'Count of used keys: {len(keys_used)}')
    print(f'Count of missing keys: {len(keys_miss)}')
    print(f'missing keys: {keys_miss}')
    return None


def setup_cfg(arg_parser):
    """
    Create configs and perform basic setups.
    """
    config = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    config.DATALOADER.NUM_WORKERS = 0
    config = add_export_config(config)
    config.merge_from_file(arg_parser.config_file)
    config.merge_from_list(arg_parser.opts)
    config.freeze()
    return config


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

    # padding around
    # l, t = pad_w // 2, pad_h // 2
    # r, b = pad_w - l, pad_h - t
    # print(f'shape:{image_tensor.shape}, padding={(l, r, t, b)}')
    # image_tensor = nn.ZeroPad2d(padding=(l, r, t, b))(image_tensor)

    return image_tensor


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
    # note: private function; subject to changes
    processed_results = []
    for results_per_image in instances:
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results


def to_bin(data_loader: DataLoader, save_path: str) -> None:
    """
    convert tensors in data_loader to .bin files saving in save_path
    """
    current_dir = os.path.dirname(__file__)
    save_path = current_dir + '/' + save_path + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        image, image_name = inputs[0]['image'], inputs[0]['file_name']
        image_name = image_name.split('/')[-1].replace('.jpg', '')
        image = single_preprocessing(image).to(torch.float32)
        image = to_numpy(image.unsqueeze(0))
        image.tofile(f'{save_path}{image_name}.bin')  # save image to bin file

