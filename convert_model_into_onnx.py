# This is a script for converting CenterMask model into ONNX format
# it will do some modification to the generate ONNX model as well
# Author: zengyan
# Final: 21.09.12

import os
import torch
import argparse
import sys
import onnx
sys.path.append('./centermask2')

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from modified_class import GeneralizedRCNN
from deploy_utils import (check_keys, setup_cfg, get_sample_inputs, single_preprocessing,
                          single_wrap_outputs, postprocess, single_flatten_to_tuple)


if __name__ == "__main__":
    '''
    run this file like:
    python convert_model_into_onnx.py --config-file "centermask2/configs/centermask/zy_model_config.yaml"  --version 11 --verbose-on \
    MODEL.WEIGHTS "centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # modify forward function of model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('\n' * 5, 'USING MODIFIED META ARCHITECTURE')

    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--pic-file", default="", metavar="FILE", help="path to pic file")
    parser.add_argument("--version", default=11, type=int, help="version of operator set")
    parser.add_argument("--forward", action="store_true", help="doing model inference to show the outputs")
    parser.add_argument("--verbose-on", action="store_true", help="show verbose")
    parser.add_argument("--fix-k", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = setup_cfg(args)
    print(cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.PROPOSAL_GENERATOR.NAME, cfg.MODEL.ROI_HEADS.NAME)
    print('\n' * 5)

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
    # print(model)

    # forward
    if args.forward:
        with torch.no_grad():
            outputs = model(inputs)
            print('\n' * 5, f'shapes of model outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

        # postprocessing
        outputs = single_wrap_outputs(outputs)
        outputs = postprocess(outputs)  # [{'instances':}]
        outputs = single_flatten_to_tuple(outputs[0]['instances'])
        print('\n' * 5, f'shapes of post processed outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

    # convert into onnx
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
    print('Converting model into ONNX ... ')
    torch.onnx.export(model, inputs, onnx_path,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                      opset_version=args.version, verbose=args.verbose_on)

    # print('Modify the structure of ONNX ... ')
    # model = onnx.load(onnx_path)
    #
    # print('Modify slice nodes ... ')
    # lst_node_idx = []
    # for idx, n in enumerate(model.graph.node):
    #     if (n.name.find('Slice') != -1) and (model.graph.node[idx + 1].name.find('Cast') != -1):
    #         node_id = idx  # 节点在onnx中真实的索引
    #         lst_node_idx.append(node_id)
    # lst_node_idx = lst_node_idx[0:6]
    # print(lst_node_idx)
    #
    # lst_drop = []
    # for idx in lst_node_idx:
    #     slice_node = model.graph.node[idx]  # 按索引获取节点
    #     lst_drop.append(slice_node)
    #     # print(f'\nInputs of node {idx} ({slice_node.name}): ', slice_node.input)
    #     # print(f'Outputs of node {idx} ({slice_node.name}): ', slice_node.output)
    #     cast_idx = idx + 1
    #     cast_node = model.graph.node[idx + 1]
    #     cast_new = onnx.helper.make_node(
    #         'Cast',
    #         name=f'Cast_m_{cast_idx}',
    #         inputs=[slice_node.input[0]],
    #         outputs=cast_node.output,
    #         to=7
    #     )  # 创建新节点
    #
    #     model.graph.node.remove(cast_node)  # 删除原节点
    #     model.graph.node.insert(cast_idx, cast_new)  # 添加新节点
    #     # print(f'New node {cast_idx}:')
    #     # print(model.graph.node[cast_idx])
    # for node in lst_drop:
    #     model.graph.node.remove(node)
    #
    # if args.fix_k:
    #     print('Saving ONNX ... ')
    #     onnx.save(model, onnx_path)
    #     print('All Done.')
    #     exit('No further Modification for ONNX so the K of TopK is fixed ... ')
    #
    # print('Modify TopK nodes ... ')
    # lst_node_idx = []
    # for idx, n in enumerate(model.graph.node):
    #     if n.name.find('TopK') != -1:
    #         node_id = idx  # 节点在onnx中真实的索引
    #         lst_node_idx.append(node_id)
    # print(lst_node_idx)
    #
    # for idx in lst_node_idx:
    #     node = model.graph.node[idx]  # 按索引获取节点
    #     # print(f'\nInputs of node {idx} ({node.name}): ', node.input)
    #     # print(f'Outputs of node {idx} ({node.name}): ', node.output)
    #     inputs_new = [node.input[0], model.graph.node[idx - 1].output[0]]  # 修改输入
    #     outputs_new = node.output  # 修改输出
    #     topk_new = onnx.helper.make_node(
    #         'TopK',
    #         name=f'TopK_m_{idx}',
    #         inputs=inputs_new,
    #         outputs=outputs_new
    #     )  # 创建新节点
    #
    #     model.graph.node.remove(node)  # 删除原节点
    #     model.graph.node.insert(idx, topk_new)  # 添加新节点
    #     # print(f'New node {idx}:')
    #     # print(model.graph.node[idx])
    #
    # # onnx.checker.check_model(model)  # 替换自定义算子后check无法通过
    # print('Saving ONNX ... ')
    # onnx.save(model, onnx_path)
    # print('All Done.')
