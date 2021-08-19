# import onnxruntime
# import torch
#
# from deploy_utils import get_sample_inputs, single_preprocessing
#
#
# def to_numpy(tensor: torch.Tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
#
# if __name__ == '__main__':
#     '''
#     run this file like:
#     python eval_onnx.py
#     '''
#     # set path
#     img_path = '000000000001.jpg'
#     onnx_path = 'centermask2.onnx'
#
#     # data processing
#     img = get_sample_inputs(img_path)
#     img = single_preprocessing(img)
#     img = img.unsqueeze(0)
#     print(f'Image: {img_path}, Shape: {img.shape}')
#     img = to_numpy(img)
#
#     # create session
#     onnx_session = onnxruntime.InferenceSession(onnx_path)
#
#     # inference
#     lst_output_nodes = [node.name for node in onnx_session.get_outputs()]
#     input_node = [node.name for node in onnx_session.get_inputs()][0]
#     outputs = onnx_session.run(lst_output_nodes, {input_node: img})
#
#     # TODO: post processing
#     # output = np.array(outputs)
#
#     print(outputs)
#     print([i.shape for i in outputs])


import torch


def choose_k(mat: torch.tensor) -> int:
    c = torch.sum(mat)
    k = 50 if c > 50 else 5
    return k


per_candidate_inds = torch.tensor([[1, 1, 1, 0],
                                   [0, 2, 0, 1],
                                   [0, 0, 0, 0],
                                   [1, 3, 1, 0]])

# a =
# k = torch.min()
a = torch.tensor([3], dtype=torch.float)
b = torch.tensor(9, dtype=torch.float)
c = torch.min(a, b)
print(a[0])
print(a.shape, b.reshape(1).shape, c.shape)






