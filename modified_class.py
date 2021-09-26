# defining some simplified classes using in deploying to replace those in detectron2
# Author: zengyan
# Final: 21.09.12

import torch
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN as RCNN
from deploy_utils import single_flatten_to_tuple


class FakeImageList(object):
    def __init__(self, tensor: torch.Tensor, hw=None):
        """
        伪造的detectron2中的ImageList类，只提供模型推理会使用到的len()和image_sizes
        :param tensor: Tensor of shape (N, H, W)
        image_sizes (list[tuple[H, W]]): Each tuple is (h, w). It can be smaller than (H, W) due to padding.
        """
        if hw is None:
            self.image_sizes = [(1344, 1344) for _ in range(tensor.shape[0])]
        else:
            self.image_sizes = hw
        self.tensor = tensor

    def __len__(self) -> int:
        return len(self.image_sizes)


class GeneralizedRCNN(RCNN):
    def forward(self, img_tensors: torch.Tensor) -> tuple:
        """
        A simplified GeneralizedRCNN for converting pth into onnx,
        without processing (such as preprocessing and postprocessing) and branches not used in inference
        """
        assert not self.training

        features = self.backbone(img_tensors)
        images = FakeImageList(img_tensors)
        proposals, _ = self.proposal_generator(images, features, None)  # Instance[pred_boxes, scores, pred_classes, locations]
        results, _ = self.roi_heads(images, features, proposals, None)
        results = single_flatten_to_tuple(results[0])
        return results
