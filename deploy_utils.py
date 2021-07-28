import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import detectron2.data.transforms as T
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Instances, Boxes
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import add_export_config
from centermask.config import get_cfg

MIN_EDGE_SIZE = 800
MAX_EDGE_SIZE = 1333
FIXED_EDGE_SIZE = 1344


def to_numpy(tensor: torch.Tensor):
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
    从路径读取一张图片，并按最短边缩放到800，且最长边不超过1333，输出为[{"image": tensor, "height": tensor, "width": tensor}]
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
    l, t = pad_w // 2, pad_h // 2
    r, b = pad_w - l, pad_h - t
    print(f'shape:{image_tensor.shape}, padding={(l, r, t, b)}')
    image_tensor = nn.ZeroPad2d(padding=(l, r, t, b))(image_tensor)

    return image_tensor


def single_wrap_outputs(tuple_outputs: tuple, height=MAX_EDGE_SIZE, width=MAX_EDGE_SIZE) -> list:
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


def single_flatten_to_tuple(wrapped_outputs: object):
    """
    模型输出被[Instances.fields]的形式封装，将不同的输出拆解出来组成元组
    :param wrapped_outputs:
    :return:
    """
    field = wrapped_outputs.get_fields()
    tuple_outputs = (field['locations'], field['mask_scores'],
                     field['pred_boxes'].tensor, field['pred_classes'],
                     field['pred_masks'], field['scores'])
    return tuple_outputs


def postprocess(instances: list, height=MAX_EDGE_SIZE, width=MAX_EDGE_SIZE, padded=False):
    """
    Rescale the output instances to the target size.
    Instances.fields:
        需处理 locations:    [N, 2]
        不处理 mask_scores:  [N, 1]
        需处理 pred_boxes:   [N, 4]
        不处理 pred_classes: [N, 1]
        需处理 pred_masks:   [N, 1, 28, 28]
        不处理 scores:       [N, 1]

    :param instances: list[Instances]
    :param height: list[dict[str, torch.Tensor]]
    :param width
    :param padded
    :return:
    """
    # note: private function; subject to changes
    processed_results = []
    for results_per_image in instances:
        if padded:
            locations = results_per_image.get_fields()['locations']
            print('\n' * 5, results_per_image.get_fields()['pred_boxes'])
            print('\n' * 5, results_per_image.get_fields()['pred_masks'].shape)
            locations = postprocess_locations(results_per_image.get_fields()['locations'], (height, width))

            results_per_image.get_fields()['locations'] = locations
            print('\n' * 5, results_per_image.get_fields()['locations'])

            results_per_image.get_fields()['pred_boxes'] = \
                postprocess_boxes(results_per_image.get_fields()['pred_boxes'], (height, width))
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results


def postprocess_locations(locations: torch.Tensor, ori_sizes: iter) -> torch.Tensor:
    pad_h = FIXED_EDGE_SIZE - ori_sizes[0]
    pad_w = FIXED_EDGE_SIZE - ori_sizes[1]
    l, t = pad_w // 2, pad_h // 2
    r, b = pad_w - l, pad_h - t
    locations = locations - torch.tensor((b, l))

    return locations


def postprocess_boxes(boxes: object, ori_sizes: iter) -> torch.Tensor:
    pad_h = FIXED_EDGE_SIZE - ori_sizes[0]
    pad_w = FIXED_EDGE_SIZE - ori_sizes[1]
    l, t = pad_w // 2, pad_h // 2
    r, b = pad_w - l, pad_h - t
    boxes.tensor = boxes.tensor - torch.tensor((b, l, b, l))

    return boxes


def postprocess_bboxes(bboxes, image_size):
    org_w = image_size[0]
    org_h = image_size[1]

    scale = 800 / min(org_w, org_h)
    new_w = int(np.floor(org_w * scale))
    new_h = int(np.floor(org_h * scale))
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w) * scale

    bboxes[:, 0] = (bboxes[:, 0]) / scale
    bboxes[:, 1] = (bboxes[:, 1]) / scale
    bboxes[:, 2] = (bboxes[:, 2]) / scale
    bboxes[:, 3] = (bboxes[:, 3]) / scale

    return bboxes


def postprocess_masks(masks, image_size, net_input_width, net_input_height):
    org_w = image_size[0]
    org_h = image_size[1]

    scale = 800 / min(org_w, org_h)
    new_w = int(np.floor(org_w * scale))
    new_h = int(np.floor(org_h * scale))
    if max(new_h, new_w) > 1333:
        scale = 1333 / max(new_h, new_w) * scale

    pad_w = net_input_width - org_w * scale
    pad_h = net_input_height - org_h * scale
    top = 0
    left = 0
    hs = int(net_input_height - pad_h)
    ws = int(net_input_width - pad_w)

    masks = masks.to(dtype=torch.float32)
    res_append = torch.zeros(0, org_h, org_w)
    if torch.cuda.is_available():
        res_append = res_append.to(device='cuda')
    for i in range(masks.size(0)):
        mask = masks[i][0][top:hs, left:ws]
        mask = mask.expand((1, 1, mask.size(0), mask.size(1)))
        mask = F.interpolate(mask, size=(int(org_h), int(org_w)), mode='bilinear', align_corners=False)
        mask = mask[0][0]
        mask = mask.unsqueeze(0)
        res_append = torch.cat((res_append, mask))

    return res_append[:, None]


