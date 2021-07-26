import torch
import torch.nn as nn
import argparse
import onnxruntime

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
from pth_to_onnx import GeneralizedRCNN, FakeImageList
from modified_class import GeneralizedRCNN as MRCNN


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


def cmp(a, b):
    n = len(a)
    return [torch.abs(a[i].to(torch.float32)-b[i].to(torch.float32)).sum() for i in range(n)]


if __name__ == "__main__":
    '''
    run this file like:
    python compare_output.py --config-file "configs/centermask/zy_model_config.yaml" --pic-file "000000000139.jpg" \
    MODEL.WEIGHTS "/export/home/zy/centermask2/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--pic-file", default="", metavar="FILE", help="path to pic file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    # print(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    # print(cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.PROPOSAL_GENERATOR.NAME)
    print('\n' * 5)

    # get a batch from given pic
    img_path = args.pic_file
    batched_inputs = get_sample_inputs(img_path)

    # pre processing
    inputs = single_preprocessing(batched_inputs[0]['image'])
    inputs = inputs.unsqueeze(0)

    # build origin model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(MRCNN)  # re-registry RCNN
    print('USING MODIFIED META ARCHITECTURE (inference)')
    origin_model = build_model(cfg)
    DetectionCheckpointer(origin_model).load(cfg.MODEL.WEIGHTS)  # load weights
    origin_model.eval()

    # build modified model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('USING MODIFIED META ARCHITECTURE (forward)')
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).load(cfg.MODEL.WEIGHTS)  # load weights
    torch_model.eval()

    # fix input compare model output
    with torch.no_grad():
        img_lst = FakeImageList(inputs, [(batched_inputs[0]['height'], batched_inputs[0]['width'])])
        outputs1 = origin_model.inference(img_lst, do_preprocess=False, do_postprocess=False)
        outputs2 = torch_model(inputs)
        o1 = single_flatten_to_tuple(outputs1[0])
        o2 = outputs2
        print(f'model outputs of fixed inputs:\n{cmp(o1, o2)}\n')

    # compare post processing
    o1 = origin_model._postprocess(outputs1, batched_inputs, img_lst.image_sizes)
    o2 = postprocess(outputs1, batched_inputs[0]['height'], batched_inputs[0]['width'])  # [{'instances':}]
    o1 = single_flatten_to_tuple(o1[0]['instances'])
    o2 = single_flatten_to_tuple(o2[0]['instances'])
    print(f'process outputs of fixed inputs:\n{cmp(o1, o2)}\n')


    # forward
    # with torch.no_grad():
    #     batched_inputs = [{"image": inputs.squeeze(0), "height": batched_inputs[0]['height'], "width": batched_inputs[0]['width']}]
    #     outputs1 = origin_model.inference(batched_inputs, do_postprocess=False)
    #     outputs2 = torch_model(inputs)
    #
    #     o1 = single_flatten_to_tuple(outputs1[0])
    #     o2 = outputs2
    #     # print(f'before processing:\n {o1[4]} \n {o2[4]}\n')
    #     print(f'{o1[4].shape}, {o2[4].shape}\n {(o1[4]-o2[4]).sum()}')
    #     # print('\n' * 5, f'shapes of model outputs:\n {[i.shape for i in outputs]}', '\n' * 5)
    #
    # # postprocessing
    # outputs1 = origin_model._postprocess(outputs1, batched_inputs, origin_model.preprocess_image(batched_inputs).image_sizes)
    # outputs2 = single_wrap_outputs(outputs2)
    # outputs2 = postprocess(outputs2, batched_inputs[0]['height'], batched_inputs[0]['width'])  # [{'instances':}]
    #
    # o1 = single_flatten_to_tuple(outputs1[0]['instances'])
    # o2 = single_flatten_to_tuple(outputs2[0]['instances'])
    # print(f'after processing:\n {o1[4]} \n {o2[4]}\n')
    # print(f'{o1[4].shape}, {o2[4].shape}\n {(o1[4]^o2[4]).sum()}')
    # # print('\n' * 5, f'shapes of post processed outputs:\n {[i.shape for i in outputs]}', '\n' * 5)
    #
    # # build onnx model
    # onnx_path = 'centermask2.onnx'
    # onnx_session = onnxruntime.InferenceSession(onnx_path)
    # onnx_input = inputs.detach().cpu().numpy()
    # lst_output_nodes = [node.name for node in onnx_session.get_outputs()]
    # input_node = [node.name for node in onnx_session.get_inputs()][0]
    # outputs3 = onnx_session.run(lst_output_nodes, {input_node: onnx_input})
    # print(outputs3[4])



    # compare post processing

    #