import onnxruntime
import torch
import torch.nn as nn
import numpy as np

import detectron2.data.transforms as T

from detectron2.data import detection_utils


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


def get_sample_inputs(path: str) -> torch.Tensor:
    # load image from path and do preprocessing
    original_image = detection_utils.read_image(path, format="BGR")
    # Do same preprocessing as DefaultPredictor
    aug = T.ResizeShortestEdge([800, 800], 1333)  # [800, 800], 1333
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    # image.type(torch.uint8)
    return image


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    # set path
    img_path = '000000000016.jpg'
    onnx_path = 'centermask2.onnx'

    # data processing
    img = get_sample_inputs(img_path)
    img = single_preprocessing(img)
    img = img.unsqueeze(0)
    print(f'Image: {img_path}, Shape: {img.shape}')
    img = to_numpy(img)

    # create session
    onnx_session = onnxruntime.InferenceSession(onnx_path)

    # inference
    lst_output_nodes = [node.name for node in onnx_session.get_outputs()]
    input_node = [node.name for node in onnx_session.get_inputs()][0]
    outputs = onnx_session.run(lst_output_nodes, {input_node: img})

    # TODO: post processing
    # output = np.array(outputs)

    print(outputs)
    print([i.shape for i in outputs])