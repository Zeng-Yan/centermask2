import torch
import torch.nn as nn
import argparse
import onnxruntime
from collections import OrderedDict

from centermask.config import get_cfg
from centermask.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import print_csv_format
from detectron2.export import add_export_config
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Instances, Boxes


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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


def single_preprocessing(image_tensor: torch.Tensor) -> torch.Tensor:
    """
        Normalize and pad the input images.
    """
    # Normalize
    pixel_mean = torch.tensor([103.53, 116.28, 123.675]).view(-1, 1, 1)
    pixel_std = torch.tensor([1.0, 1.0, 1.0]).view(-1, 1, 1)
    image_tensor = (image_tensor - pixel_mean) / pixel_std

    # Padding
    pad_h = 1344 - image_tensor.shape[1]
    pad_w = 1344 - image_tensor.shape[2]
    l, t = pad_w // 2, pad_h // 2
    r, b = pad_w - l, pad_h - t
    print(f'shape:{image_tensor.shape}, padding={(l, r, t, b)}')
    image_tensor = nn.ZeroPad2d(padding=(l, r, t, b))(image_tensor)

    return image_tensor


def single_wrap_outputs(tuple_outputs: tuple) -> list:
    """
    将元组形式的模型输出重新封装成[Instances.fields]的形式
    :param tuple_outputs:
    :return:
    """
    instances = Instances((1333, 1333))
    tuple_outputs = [torch.tensor(x)[:50] for x in tuple_outputs]
    instances.set('locations', tuple_outputs[0])
    instances.set('mask_scores', tuple_outputs[1])
    instances.set('pred_boxes', Boxes(tuple_outputs[2]))
    instances.set('pred_classes', tuple_outputs[3])
    instances.set('pred_masks', tuple_outputs[4])
    instances.set('scores', tuple_outputs[5])

    return [instances]


def postprocess(instances: list, height, width):
    """
    Rescale the output instances to the target size.
    :param instances: list[Instances]
    :param batched_inputs: list[dict[str, torch.Tensor]]
    :return:
    """
    # note: private function; subject to changes
    processed_results = []
    for results_per_image in instances:
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results


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


def inference_on_dataset(session, data_loader, evaluator):
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


def test(session, config):
    results = OrderedDict()
    for idx, dataset_name in enumerate(config.DATASETS.TEST):
        data_loader = build_detection_test_loader(config, dataset_name)
        evaluator = COCOEvaluator(dataset_name, output_dir=config.OUTPUT_DIR)
        results_i = inference_on_dataset(session, data_loader, evaluator)
        results[dataset_name] = results_i
        print_csv_format(results_i)

    if len(results) == 1:
        results = list(results.values())[0]
    return results


if __name__ == '__main__':
    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # build onnx model
    onnx_path = 'centermask2.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_path)

    # inference

    # test on onnx model
    test(onnx_session, cfg)
