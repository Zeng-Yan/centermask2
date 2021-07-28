import torch
import torch.nn as nn
import argparse
import onnxruntime
from collections import OrderedDict

from centermask.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import print_csv_format
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from modified_class import GeneralizedRCNN, FakeImageList
from deploy_utils import to_numpy, single_preprocessing, single_wrap_outputs, postprocess, setup_cfg


def inference_onnx(session, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        image, h, w = inputs[0]['image'], inputs[0]['height'], inputs[0]['width']
        # print('\n' * 5, h, w, inputs.shape, '\n' * 5)
        image = single_preprocessing(image).to(torch.float32)
        image = to_numpy(image.unsqueeze(0))
        lst_output_nodes = [node.name for node in session.get_outputs()]
        input_node = [node.name for node in session.get_inputs()][0]
        outputs = session.run(lst_output_nodes, {input_node: image})
        outputs = single_wrap_outputs(outputs)
        outputs = postprocess(outputs, h, w)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def inference_fixed(model, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        image, h, w = inputs[0]['image'], inputs[0]['height'], inputs[0]['width']
        # print('\n' * 5, h, w, inputs.shape, '\n' * 5)
        image = single_preprocessing(image).to(torch.float32).unsqueeze(0)
        img_lst = FakeImageList(image, [(inputs[0]['height'], inputs[0]['width'])])
        with torch.no_grad():
            outputs = model.inference(img_lst, do_preprocess=False, do_postprocess=False)
        # outputs = single_flatten_to_tuple(outputs[0])
        # outputs = (x.detach() for x in outputs)
        # outputs = single_wrap_outputs(outputs, inputs[0]['height'], inputs[0]['width'])
        outputs = model._postprocess(outputs, inputs, img_lst.image_sizes)

        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def inference_origin(model, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        with torch.no_grad():
            outputs = model(inputs)
        # outputs = single_flatten_to_tuple(outputs[0]['instances'])
        # outputs = (x.detach() for x in outputs)
        # outputs = single_wrap_outputs(outputs, inputs[0]['height'], inputs[0]['width'])
        # outputs = [{'instances': outputs[0]}]
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def test(launcher, config, typ):
    results = OrderedDict()
    for idx, dataset_name in enumerate(config.DATASETS.TEST):
        data_loader = build_detection_test_loader(config, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=config.OUTPUT_DIR)
        if typ == 'onnx':
            results_i = inference_onnx(launcher, data_loader, evaluator)
        elif typ == 'fixed':
            results_i = inference_fixed(launcher, data_loader, evaluator)
        else:
            results_i = inference_origin(launcher, data_loader, evaluator)
        results[dataset_name] = results_i
        print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]
    return results


if __name__ == '__main__':
    '''
    run this file like:
    python test.py --config-file "configs/centermask/zy_model_config.yaml" \
     --type pth MODEL.WEIGHTS "/export/home/zy/centermask2/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--type", default="onnx", help="model type")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # build onnx model
    if args.type == 'onnx':
        onnx_path = 'centermask2.onnx'
        onnx_session = onnxruntime.InferenceSession(onnx_path)
        lch = onnx_session
    elif args.type == 'fixed':
        META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
        META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
        print('USING MODIFIED META ARCHITECTURE (inference)')
        fixed_model = build_model(cfg)
        DetectionCheckpointer(fixed_model).load(cfg.MODEL.WEIGHTS)  # load weights
        fixed_model.eval()
        lch = fixed_model
    else:
        print('USING original META ARCHITECTURE')
        ori_model = build_model(cfg)
        DetectionCheckpointer(ori_model).load(cfg.MODEL.WEIGHTS)  # load weights
        ori_model.eval()
        lch = ori_model

    # evaluation
    test(lch, cfg, args.type)
