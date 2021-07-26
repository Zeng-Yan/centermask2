import torch
import torch.nn as nn
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import add_export_config
from centermask.config import get_cfg


def check_keys(model: nn.Module, state_dict: dict):
    keys_model = set(model.state_dict().keys())
    keys_state = set(state_dict.keys())
    keys_miss = keys_model - keys_state
    keys_used = keys_model & keys_state
    print(f'{len(keys_model)} keys of model, {len(keys_state)} keys of state_dict')
    print(f'Count of used keys: {len(keys_used)}')
    print(f'Count of missing keys: {len(keys_miss)}')
    print(f'missing keys: {keys_miss}')
    return 0


def setup_cfg(args):
    """
        Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_sample_inputs(path: str) -> list:
    # load image from path
    original_image = detection_utils.read_image(path, format="BGR")
    height, width = original_image.shape[:2]
    # resize
    aug = T.ResizeShortestEdge([800, 800], 1333)  # [800, 800], 1333
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    dic_img = {"image": image, "height": height, "width": width}
    return [dic_img]


