# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.data import detection_utils

from deploy_utils import setup_cfg, get_sample_inputs
from test import single_preprocessing, postprocess
from modified_class import GeneralizedRCNN


def run_on_image(predictions, image):
    """
    Args:
        predictions
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
        vis_output (VisImage): the visualized image output.
    """
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = image[:, :, ::-1]
    visualizer = Visualizer(image, cfg.DATASETS.TEST[0], instance_mode=ColorMode.IMAGE)
    instances = predictions["instances"]
    vis_output = visualizer.draw_instance_predictions(predictions=instances)
    return predictions, vis_output


if __name__ == "__main__":
    # set cfg
    parser = argparse.ArgumentParser(description="Visualizer")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--pic-file", default="", metavar="FILE", help="path to pic file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = setup_cfg(args)

    # get a batch from given pic
    img_path = args.pic_file
    batched_inputs = get_sample_inputs(img_path)
    inputs = single_preprocessing(batched_inputs[0]['image'])  # preprocessing
    inputs = inputs.unsqueeze(0)

    # build modified model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('USING MODIFIED META ARCHITECTURE (forward)')
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).load(cfg.MODEL.WEIGHTS)  # load weights
    torch_model.eval()

    # fix input compare model output
    with torch.no_grad():
        outputs = torch_model(inputs)
    outputs = postprocess(outputs, batched_inputs[0]['height'], batched_inputs[0]['width'])

    # visualize outputs
    original_image = detection_utils.read_image(img_path, format="BGR")
    pred, visualized_output = run_on_image(outputs[0], original_image)
    visualized_output.save('visualized_outputs.jpg')
