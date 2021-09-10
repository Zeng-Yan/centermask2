# do post-processing on bin-format outputs of deployed model and eval them
# Author: zengyan
# Final: 21.09.10

import argparse
import torch
import numpy as np
import sys
sys.path.append('./centermask2')

from centermask.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import print_csv_format

from deploy_utils import single_wrap_outputs, postprocess, setup_cfg


def buf_to_tensor(b: np.array, shape: tuple) -> torch.tensor:
    return torch.from_numpy(b).reshape(shape)


def eval_bin_res(arguments) -> None:
    cfg = setup_cfg(arguments)
    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    evaluator.reset()
    exceptions = []
    for idx, inputs in enumerate(data_loader):
        image_path, h, w = inputs[0]['file_name'], inputs[0]['height'], inputs[0]['width']
        iamge_name = image_path.split('/')[-1].split('.')[0]
        print(iamge_name)

        try:
            lst_res = []
            # loc, mask_s, boxes, class, mask, score
            lst_dtype = ['float32', 'float32', 'float32', 'int64', 'float32', 'float32']
            lst_shape = [(-1, 2), (-1), (-1, 4), (-1), (-1, 1, 28, 28), (-1)]
            for idx in range(6):
                bin_path = arguments.bin_data_path + iamge_name + "_" + str(idx+1) + ".bin"
                buf = np.fromfile(bin_path, dtype=lst_dtype[idx])
                # print(idx, buf.shape)
                buf = buf_to_tensor(buf, lst_shape[idx])
                lst_res.append(buf)
            outputs = single_wrap_outputs(lst_res)
            outputs = postprocess(outputs, h, w)
            evaluator.process(inputs, outputs)
        except FileNotFoundError:
            exceptions.append(iamge_name)
            print(f'Missing Pic: {iamge_name}: bin-files not exist.')

    print(f'[Warning] Missing bin-files of pics: {exceptions}')
    print_csv_format(evaluator.evaluate())

    return None


if __name__ == '__main__':
    '''
    run this file like:
    python postprocess_bin_outputs.py --config-file "centermask2/configs/centermask/zy_model_config.yaml" \
    --bin-data-path "./result/dumpOutput_device0/"
    '''
    # set cfg
    parser = argparse.ArgumentParser(description="Eval bin-outputs of inference.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--bin-data-path", default="./result/dumpOutput_device0/")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER, )
    args = parser.parse_args()

    # eval
    eval_bin_res(args)





