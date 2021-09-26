# do evaluation to results of inference in different model-format
# Author: zengyan
# Final: 21.09.12

import torch
import argparse
from onnxruntime import InferenceSession
from collections import OrderedDict
import sys
sys.path.append('./centermask2')

from centermask.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import print_csv_format
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN

from modified_class import FakeImageList
from deploy_utils import to_numpy, single_preprocessing, single_wrap_outputs, postprocess, setup_cfg


class GeneralizedRCNN(RCNN):
    def inference(
        self,
        batched_inputs,
        detected_instances=None,
        do_preprocess: bool = True,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_preprocess
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        if do_preprocess:
            images = self.preprocess_image(batched_inputs)
        else:
            images = batched_inputs

        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results


def inference_onnx(session, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        image, h, w = inputs[0]['image'], inputs[0]['height'], inputs[0]['width']
        image = single_preprocessing(image).to(torch.float32)
        image = to_numpy(image.unsqueeze(0))
        lst_output_nodes = [node.name for node in session.get_outputs()]
        input_node = [node.name for node in session.get_inputs()][0]
        outputs = session.run(lst_output_nodes, {input_node: image})
        outputs = single_wrap_outputs(outputs)
        outputs = postprocess(outputs, h, w)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def inference_mod(model, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        image, h, w = inputs[0]['image'], inputs[0]['height'], inputs[0]['width']
        image = single_preprocessing(image).to(torch.float32).unsqueeze(0)
        img_lst = FakeImageList(image, [(inputs[0]['height'], inputs[0]['width'])])
        with torch.no_grad():
            outputs = model.inference(img_lst, do_preprocess=False, do_postprocess=False)
        outputs = postprocess(outputs, h, w)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def inference_origin(model, data_loader, evaluator):
    evaluator.reset()
    for idx, inputs in enumerate(data_loader):
        print(inputs[0]['file_name'])
        with torch.no_grad():
            outputs = model(inputs)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def test(launcher, config, typ):
    results = OrderedDict()
    for idx, dataset_name in enumerate(config.DATASETS.TEST):
        data_loader = build_detection_test_loader(config, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=config.OUTPUT_DIR)
        if typ == 'onnx':
            results_i = inference_onnx(launcher, data_loader, evaluator)
        elif typ == 'mod':
            results_i = inference_mod(launcher, data_loader, evaluator)
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
    python tester.py --config-file "centermask2/configs/centermask/zy_model_config.yaml" \
     --type onnx MODEL.WEIGHTS "/home/zeng/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="eval results of inference in different model-format")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--type", default="onnx", help="model type")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # build onnx model
    if args.type == 'onnx':
        onnx_path = 'centermask2.onnx'
        onnx_session = InferenceSession(onnx_path)
        lch = onnx_session
    elif args.type == 'mod':
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
