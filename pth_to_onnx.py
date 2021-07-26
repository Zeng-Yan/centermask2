import torch
import torch.nn as nn
import argparse

import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN
from detectron2.modeling import build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.export import add_export_config
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Instances, Boxes
from centermask.config import get_cfg

from test import single_wrap_outputs, single_preprocessing, postprocess, single_flatten_to_tuple

    # lst_of_fields = [instance.fields for instance in wrapped_outputs]
    #
    # lst_of_pred_boxes = [field['pred_boxes'].tensor for field in lst_of_fields]
    # lst_of_scores = [field['scores'] for field in lst_of_fields]
    # lst_of_pred_classes = [field['pred_classes'] for field in lst_of_fields]
    # lst_of_locations = [field['locations'] for field in lst_of_fields]
    # lst_of_pred_masks = [field['pred_masks'] for field in lst_of_fields]
    # lst_of_mask_scores = [field['mask_scores'] for field in lst_of_fields]


class FakeImageList(object):
    def __init__(self, tensor: torch.Tensor, hw=None):
        """
        伪造的detectron2中的ImageList类，只提供模型推理会使用到的len()和image_sizes
        :param tensor: Tensor of shape (N, H, W)
        image_sizes (list[tuple[H, W]]): Each tuple is (h, w). It can be smaller than (H, W) due to padding.
        """
        if hw is None:
            self.image_sizes = [(1333, 1333) for _ in range(tensor.shape[0])]
        else:
            self.image_sizes = hw
        self.tensor = tensor

    def __len__(self) -> int:
        return len(self.image_sizes)


class GeneralizedRCNN(RCNN):
    def forward(self, img_tensors: torch.Tensor) -> tuple:
        """
        去掉预处理和不执行的分支，用精简后的inference替代forward，伪造FakeImageList类替代ImageList类
        """
        assert not self.training

        features = self.backbone(img_tensors)
        images = FakeImageList(img_tensors)
        proposals, _ = self.proposal_generator(images, features, None)
        # proposals: Instance.fields(pred_boxes, scores, pred_classes, locations)
        results, _ = self.roi_heads(images, features, proposals, None)
        # print(results)
        results = single_flatten_to_tuple(results[0])
        return results


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


if __name__ == "__main__":
    '''
    run this file like:
    python pth_to_onnx.py --config-file "configs/centermask/zy_model_config.yaml" --pic-file "000000000139.jpg" /
    --onnx MODEL.WEIGHTS "/export/home/zy/centermask2/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # modified forward function of model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('\n' * 5, 'USING MODIFIED META ARCHITECTURE')

    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--pic-file", default="", metavar="FILE", help="path to pic file")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--v", action="store_true")
    # parser.add_argument("--run-eval", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = setup_cfg(args)
    # print(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    print(cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.PROPOSAL_GENERATOR.NAME)
    print('\n' * 5)

    # build torch model
    model = build_model(cfg)
    # path_pth = 'centermask2-V-39-eSE-FPN-ms-3x.pth'
    path_pth = cfg.MODEL.WEIGHTS
    check_keys(model, torch.load(path_pth)['model'])  # compare keys
    DetectionCheckpointer(model).load(path_pth)  # load weights
    model.eval()
    # print(model)

    # get a batch from given pic
    img_path = args.pic_file
    batched_inputs = get_sample_inputs(img_path)

    # preprocessing
    inputs = single_preprocessing(batched_inputs[0]['image'])
    inputs = inputs.unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = model(inputs)
        print('\n' * 5, f'shapes of model outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

    # postprocessing
    outputs = single_wrap_outputs(outputs)
    outputs = postprocess(outputs, batched_inputs)  # [{'instances':}]
    outputs = single_flatten_to_tuple(outputs[0]['instances'])
    print('\n' * 5, f'shapes of post processed outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

    # convert to onnx
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
    if args.onnx:
        torch.onnx.export(model, inputs, 'centermask2.onnx',
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=11, verbose=args.v)

