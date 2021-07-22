import argparse
import onnxruntime
from collections import OrderedDict

from centermask.config import get_cfg
from centermask.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import print_csv_format
from detectron2.export import add_export_config


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


def inference_on_dataset(session, data_loader, evaluator):
    for idx, inputs in enumerate(data_loader):
        print('\n'*5, inputs, '\n'*5)
        lst_output_nodes = [node.name for node in session.get_outputs()]
        input_node = [node.name for node in session.get_inputs()][0]
        outputs = session.run(lst_output_nodes, {input_node: inputs})
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
