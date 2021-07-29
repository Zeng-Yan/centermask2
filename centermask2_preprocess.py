import argparse
import sys
sys.path.append('./centermask2')

from detectron2.data import build_detection_test_loader

from deploy_utils import setup_cfg, to_bin


if __name__ == '__main__':
    '''
    run this file like:
    python centermask2_preprocess.py --config-file "centermask2/configs/centermask/zy_model_config.yaml"
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
    cfg = setup_cfg(args)

    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    to_bin(data_loader, 'input_bins')


