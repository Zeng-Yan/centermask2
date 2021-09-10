# do pre-processing on pics and then convert them into bin-files
# Author: zengyan
# Final: 21.09.10

import argparse

from detectron2.data import build_detection_test_loader

from deploy_utils import setup_cfg, to_bin


if __name__ == '__main__':
    '''
    run this file like:
    python preprocess_inputs_to_bin.py --config-file "centermask2/configs/centermask/zy_model_config.yaml"
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="Convert input pics into bin file.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    cfg = setup_cfg(args)

    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    to_bin(data_loader, 'input_bins')


