import torch
import argparse

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from modified_class import FakeImageList
from deploy_utils import check_keys, setup_cfg, get_sample_inputs, single_wrap_outputs, single_preprocessing, postprocess, single_flatten_to_tuple
    # lst_of_fields = [instance.fields for instance in wrapped_outputs]
    #
    # lst_of_pred_boxes = [field['pred_boxes'].tensor for field in lst_of_fields]
    # lst_of_scores = [field['scores'] for field in lst_of_fields]
    # lst_of_pred_classes = [field['pred_classes'] for field in lst_of_fields]
    # lst_of_locations = [field['locations'] for field in lst_of_fields]
    # lst_of_pred_masks = [field['pred_masks'] for field in lst_of_fields]
    # lst_of_mask_scores = [field['mask_scores'] for field in lst_of_fields]


class GeneralizedRCNN(RCNN):
    def forward(self, img_tensors: torch.Tensor) -> tuple:
        """
        去掉预处理和不执行的分支，用精简后的inference替代forward，伪造FakeImageList类替代ImageList类
        """
        assert not self.training

        features = self.backbone(img_tensors)
        images = FakeImageList(img_tensors)
        proposals, _ = self.proposal_generator(images, features, None)
        # proposals: Instance.fields(pred_boxes, scores, pred_classes, locations)
        results, _ = self.roi_heads(images, features, proposals, None)
        # print(results)
        results = single_flatten_to_tuple(results[0])
        return results


if __name__ == "__main__":
    '''
    run this file like:
    python pth_to_onnx.py --config-file "configs/centermask/zy_model_config.yaml" --pic-file "000000000139.jpg" /
    --onnx MODEL.WEIGHTS "/export/home/zy/centermask2/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu
    '''
    # modified forward function of model
    META_ARCH_REGISTRY._obj_map.pop('GeneralizedRCNN')  # delete RCNN from registry
    META_ARCH_REGISTRY.register(GeneralizedRCNN)  # re-registry RCNN
    print('\n' * 5, 'USING MODIFIED META ARCHITECTURE')

    # set cfg
    parser = argparse.ArgumentParser(description="Convert a model using tracing.")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--pic-file", default="", metavar="FILE", help="path to pic file")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--v", action="store_true")
    # parser.add_argument("--run-eval", action="store_true")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = setup_cfg(args)
    # print(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD)
    print(cfg.MODEL.META_ARCHITECTURE, cfg.MODEL.ROI_HEADS.NAME, cfg.MODEL.PROPOSAL_GENERATOR.NAME)
    print('\n' * 5)

    # build torch model
    model = build_model(cfg)
    # path_pth = 'centermask2-V-39-eSE-FPN-ms-3x.pth'
    path_pth = cfg.MODEL.WEIGHTS
    check_keys(model, torch.load(path_pth)['model'])  # compare keys
    DetectionCheckpointer(model).load(path_pth)  # load weights
    model.eval()
    # print(model)

    # get a batch from given pic
    img_path = args.pic_file
    batched_inputs = get_sample_inputs(img_path)

    # preprocessing
    inputs = single_preprocessing(batched_inputs[0]['image'])
    inputs = inputs.unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = model(inputs)
        print('\n' * 5, f'shapes of model outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

    # postprocessing
    outputs = single_wrap_outputs(outputs)
    outputs = postprocess(outputs, batched_inputs)  # [{'instances':}]
    outputs = single_flatten_to_tuple(outputs[0]['instances'])
    print('\n' * 5, f'shapes of post processed outputs:\n {[i.shape for i in outputs]}', '\n' * 5)

    # convert to onnx
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
    if args.onnx:
        torch.onnx.export(model, inputs, 'centermask2.onnx',
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=11, verbose=args.v)

