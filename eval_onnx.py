# refer: onnx/docs/PythonAPIOverview.md
# https://github.com/onnx/onnx/blob/f2daca5e9b9315a2034da61c662d2a7ac28a9488/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model

import torch
import onnx
import onnxruntime
import numpy as np
from onnx import helper, shape_inference, version_converter
from onnx import AttributeProto, TensorProto, GraphProto


def make_NonZeroByTopK(x: str, top: str) -> (list, list):
    """
        create onnx nodes to replace 2D-NonZero() via TopK() due to the bad support of NonZero() in ATC tools of Huawei
    """
    # compute k
    node_sum = helper.make_node('ReduceSum', [x], ['input_sum'], keepdims=0)  # 左右加
    node_cast_top = helper.make_node('Cast', [top], ['top_f'], to=1)  # to float
    node_cast_sum = helper.make_node('Cast', ['input_sum'], ['input_sum_f'], to=1)  # to float
    node_min = helper.make_node('Min', ['input_sum_f', 'top_f'], ['k_f'])  # min(x.sum(), top)
    # node_reshape = helper.make_node('Reshape', ['k_f', constant_1.name], ['k_'])  #
    node_cast_k = helper.make_node('Cast', ['k_f'], ['k'], to=7)  # to int64

    # top k of x
    node_flatten_x = helper.make_node('Flatten', [x], ['input_flatten'], axis=0)  # flatten x
    node_topk = helper.make_node('TopK', ['input_flatten', 'k'], ['k_values', 'k_idx_'], axis=1)  # top k of x
    node_cast_flatten = helper.make_node('Cast', ['k_idx_'], ['k_idx'], to=7)  # to int64

    # compute idx
    constant_1 = helper.make_tensor('c_1', onnx.TensorProto.INT64, [1], [1])  # [1]
    node_shape = helper.make_node('Shape', [x], ['input_shape'])  # x.shape
    node_shape_col = helper.make_node('Gather', ['input_shape', constant_1.name], ['input_col'], axis=0)  # x.shape[1]
    node_div = helper.make_node('Div', ['k_idx', 'input_col'], ['row'])  # int64: div(int64, int64)
    node_mod = helper.make_node('Mod', ['k_idx', 'input_col'], ['col'], fmod=0)

    # stack idx
    node_unsqueeze_r = helper.make_node('Unsqueeze', ['row'], ['row_'], axes=[-1])
    node_unsqueeze_c = helper.make_node('Unsqueeze', ['col'], ['col_'], axes=[-1])
    node_concat = helper.make_node('Concat', ['row_', 'col_'], ['y_'], axis=-1)
    node_squeez = helper.make_node('Squeeze', ['y_'], ['y'], axes=[0])

    nodes = [node_sum, node_cast_sum, node_cast_top, node_min, node_cast_k,
             node_flatten_x, node_topk, node_cast_flatten,
             node_shape, node_shape_col, node_div, node_mod,
             node_unsqueeze_r, node_unsqueeze_c, node_concat, node_squeez]
    init = [constant_1]
    return nodes, init


if __name__ == '__main__':
    '''
    testing if the output of NonZeroByTopK() match the output of torch.nonzero() with same input
    '''

    N = 3
    # create a onnx model
    # Proto details: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    X = helper.make_tensor_value_info('x', TensorProto.INT64, [N, 2])  # Create one input (ValueInfoProto)
    TOP = helper.make_tensor_value_info('top', TensorProto.INT64, [1])  # Create one input (ValueInfoProto)
    C = helper.make_tensor_value_info('c_1', TensorProto.INT64, [1])  # Create one input (ValueInfoProto)
    Y = helper.make_tensor_value_info('y', TensorProto.INT64, [-1, 2])  # Create one output (ValueInfoProto)

    lst_nodes, lst_init = make_NonZeroByTopK('x', 'top')  # make nodes and initializer
    graph_def = helper.make_graph(nodes=lst_nodes, initializer=lst_init, name='NonzeroByTopK',
                                  inputs=[X, TOP, C], outputs=[Y])  # Create the graph (GraphProto)

    model_def = helper.make_model(graph_def, producer_name='nonzero')  # Create the model (ModelProto)
    model_def.opset_import[0].version = 11  # specify version
    model_def.ir_version = 7  # specify version

    onnx.checker.check_model(model_def)  # check model
    onnx.save(model_def, 'NonzeroByTopK.onnx')  # save model
    # model_onnx = onnx.load_model('NonzeroByTopK.onnx')
    # inferred_model = shape_inference.infer_shapes(model_def)
    # print(inferred_model.graph.value_info)

    # test model
    input_tensor = torch.randint(0, 2, (N, 2))
    print(f'\ninput: \n{input_tensor.numpy()}')
    print(f'\nNonzero() of input: \n{torch.nonzero(input_tensor).numpy()}')
    input_tensor = input_tensor.detach().cpu().numpy()
    top = torch.tensor([10], dtype=torch.int64).detach().cpu().numpy()
    onnx_session = onnxruntime.InferenceSession('NonzeroByTopK.onnx')
    y = onnx_session.run(['y'], {'x': input_tensor, 'top': top})
    print(f'\nNonzeroByTopK() of input: \n{y[0]}')

# 坑点1 把常量加入input
# 坑点2 修改ir_verion 和opset version
