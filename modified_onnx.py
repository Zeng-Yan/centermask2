import onnx

onnx_path = 'C:/Users/windf/Desktop/centermask2.onnx'
print('Modified ONNX')
model = onnx.load(onnx_path)

lst_node_idx = []

for idx, n in enumerate(model.graph.node):
    if n.name.find('TopK') != -1:
        print(n.name)
        node_id = idx  # 节点在onnx中真实的索引
        lst_node_idx.append(node_id)
        print(model.graph.node[node_id-1].name, model.graph.node[node_id+1].name)


print(lst_node_idx)

for idx in lst_node_idx:
    node = model.graph.node[idx]

    print(f'\nInputs of node {idx} ({node.name}): ', node.input)
    print(f'Outputs of node {idx} ({node.name}): ', node.output)
    inputs_new = [node.input[0], model.graph.node[idx-1].output[0]]
    outputs_new = node.output

    topk_new = onnx.helper.make_node(
        'TopK',
        name=f'TopK_m_{idx}',
        inputs=inputs_new,
        outputs=outputs_new
    )

    model.graph.node.remove(node)
    model.graph.node.insert(idx, topk_new)
    print(f'New node {idx}:')
    print(model.graph.node[idx])



# topk_new = onnx.helper.make_node(
#     'TopK',
#     name=f'{node_name}_m',
#     inputs=['cast_for_159_output', '494'],
#     outputs=model.graph.node[node_id].output
# )






# sizes1 = onnx.helper.make_tensor('size_230', onnx.TensorProto.INT32, [4], [1, 256, 84, 84])
# sizes2 = onnx.helper.make_tensor('size_235', onnx.TensorProto.INT32, [4], [1, 256, 168, 168])
# model.graph.initializer.append(sizes1)
# model.graph.initializer.append(sizes2)
# model.graph.node[230].input[3] = "size_230"
# getNodeByName(model_nodes, 'Resize_141').input[3] = "size_230"
# getNodeByName(model_nodes, 'Resize_161').input[3] = "size_235"
#
# print('Saving ONNX')
# onnx.save(model, onnx_path)
# print('All Done')
#
# Resize_230
# Resize_235