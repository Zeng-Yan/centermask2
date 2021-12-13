# This is a script for comparing outputs of onnx-node and torch-graph-node
# Author: zengyan
# Final: 21.09.12

import os
import torch
import argparse
import sys
import onnx
from onnx import helper, TensorProto
sys.path.append('./centermask2')

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from modified_class import GeneralizedRCNN
from deploy_utils import setup_cfg, get_sample_inputs, single_preprocessing

TENSOR_TYPE_SWITCHER = {
    'int64': TensorProto.INT64,
    'float': TensorProto.FLOAT,
}

HOOK_RECODER = {}


def get_layer_info(name):
    def hook(module, x, y):
        HOOK_RECODER[name] = (x, y)
    return hook


def get_module_outputs(args):

    # modify forward function of model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('\n' * 5, 'USING MODIFIED META ARCHITECTURE')
    cfg = setup_cfg(args)

    # get inputs
    if args.pic_file:
        img_path = args.pic_file
    else:
        img_path = os.environ['DETECTRON2_DATASETS'] + '/coco/val2017/000000000139.jpg'
    batched_inputs = get_sample_inputs(img_path)  # read and resize
    inputs = single_preprocessing(batched_inputs[0]['image']).unsqueeze(0)  # preprocessing

    # build online torch model
    model = build_model(cfg)
    path_pth = cfg.MODEL.WEIGHTS
    DetectionCheckpointer(model).load(path_pth)  # load weights
    model.eval()

    # set hooks to specified module
    for module_name, module in model.named_modules():
        # print(module_name)
        if module_name == args.module:
            print(f'\n\nHere\'s Johnny ！！！{module_name} \n\n')
            module.register_forward_hook(get_layer_info(module_name))

    with torch.no_grad():
        outputs = model(inputs)

    print('\n'*2, f'HOOK UP {args.module}: ')
    print(HOOK_RECODER)


def cut_onnx(args):
    # load model
    model = onnx.load('centermask2.onnx')

    # remove nodes after specified node
    # flag = None
    # for idx, n in enumerate(model.graph.node):
    #     if flag:
    #         model.graph.node.remove(n)
    #     if n.name == args.node:
    #         flag = True

    # create tensor-info of output
    tensor_shape = [int(x) for x in args.shape.split(',')]
    new_output_tensor_info = onnx.helper.make_tensor_value_info(
        args.tensor, TENSOR_TYPE_SWITCHER[args.type], tensor_shape)

    # remove all outputs-info then append new info
    out = [i for i in model.graph.output]
    for i in out:
        model.graph.output.remove(i)  # TODO: need a better way
    model.graph.output.append(new_output_tensor_info)

    print(model.graph.output)

    onnx.save_model(model, 'sub.onnx')

    return 0


if __name__ == "__main__":
    '''
    run this file like:
    python check_layers_outputs.py --config-file "centermask2/configs/centermask/zy_model_config.yaml" \
    --module roi_heads.mask_pooler.level_poolers.0 --node RoiExtractor_1734 --tensor 2854 --type float --shape 50,256,14,14 \
    MODEL.WEIGHTS "/home/zeng/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    print('[WARNING] after running this script, the original ONNX will be replaced')

    # set cfg
    parser = argparse.ArgumentParser(description="get outputs and onnx of a sub-model")
    parser.add_argument("--config-file", required=True,  default="",      help="path to config file", metavar="FILE",)
    parser.add_argument("--pic-file",    required=False, default="",      help="path to pic file", metavar="FILE",)
    parser.add_argument("--module",      required=False, default="",      help="name of module")
    parser.add_argument("--node",        required=False, default="",      help="name of onnx node")
    parser.add_argument("--tensor",      required=False, default="",      help="name of onnx tensor")
    parser.add_argument("--type",        required=False, default='float', help="type of onnx tensor", choices=['int64', 'float'], )
    parser.add_argument("--shape",       required=False, default="1",     help="shape of onnx tensor")
    parser.add_argument("opts", default=None, help="Modify config options using the command-line", nargs=argparse.REMAINDER,)
    arguments = parser.parse_args()

    if arguments.module:
        get_module_outputs(arguments)
    if arguments.node:
        cut_onnx(arguments)




# 在线模型forward指定代码段输出，和om推理输出的比较
# 获取onnx的中间节点输出，去掉后面所有的节点和输出
# onnx转om，运行结果
# 在forward指定代码处break trace，按named parameters的grad判断执行到的module



