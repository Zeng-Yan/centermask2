import torch
import onnx
import onnxruntime


def nonzero(**kwargs) -> torch.tensor:
    """
    由于华为方不支持nozero算子，该函数用topk规避nonzero
    """
    if not kwargs['m']:  # 正常执行nonzero算子
        idx = torch.nonzero(kwargs['input'], )
    else:  # 导出onnx时执行的分支
        x = kwargs['input'].to(torch.float32)  # bool/? -> int64 统一数据类型避免奇怪的错误
        k = torch.sum(x)  # 设置k值
        if x.ndim == 1:
            k = torch.min(torch.tensor(x.shape[0]).to(torch.float32), k)  # 一维情况下避免索引越界，op 11 要求min为float
            k = k.reshape(1).to(torch.int64)  # topk的k必须是1d int64
            _, idx = x.topk(k.item())
            idx = idx.unsqueeze(-1)  # [M, 1] 改变形状对应nonzero的输出
        else:  # 输入为二维情况下的执行分支
            fixed_dim = torch.tensor(x.shape[1], dtype=torch.int64)  # [80] 记录固定的列数，以便还原二维索引
            x = x.flatten()  # [N, 80] -> [N*80]  奇怪的地方，这里被onnx换成了reshape
            nms_top = kwargs['nms_top']  # nms_top仅在二维情况下生效
            k = torch.min(nms_top.to(torch.float32), k)  # op 11 要求min为float
            k = k.reshape(1).to(torch.int64)  # topk的k必须是1d int64
            _, col_idx = x.topk(k.item())  # 将二维tensor展平后用topk规避
            col_idx = col_idx.to(torch.int64)  # 增加cast便于onnx修改算子
            row_idx = (col_idx / fixed_dim).floor().to(torch.int64)  # topk在原二维tensor对应的行索引
            # col_idx = col_idx.fmod(fixed_dim).to(torch.int64)  # topk在原二维tensor对应的列索引
            col_idx = (col_idx - row_idx * fixed_dim).to(torch.int64)  # opset9 不支持fmod
            idx = torch.stack((row_idx, col_idx), dim=-1)  # [k, 2] 合并为[[行索引, 列索引], ...]的形式
        idx = idx[0:k[0]]  # 一个无意义的操作来保留onnx中k的计算

    return idx.to(torch.int64)


class TestNet(torch.nn.Module):
    def __init__(self, m):
        super(TestNet, self).__init__()
        self.m = m

    def forward(self, inp):
        return nonzero(input=inp, as_tuple=False, nms_top=torch.tensor(10), m=self.m)


if __name__ == '__main__':
    net1 = TestNet(False)
    net2 = TestNet(True)
    x = torch.randint(0, 2, (3, 2))
    print(x)
    x_array = x.detach().cpu().numpy()

    onnx_path = 'Net1.onnx'
    torch.onnx.export(net1, x, onnx_path,
                      input_names=['x'], output_names=['y'], dynamic_axes={'y': [0]},
                      opset_version=11, verbose=False)

    onnx_session = onnxruntime.InferenceSession(onnx_path)
    output = onnx_session.run(['y'], {'x': x_array})
    print(output)

    onnx_path = 'Net2.onnx'
    torch.onnx.export(net2, x, onnx_path,
                      input_names=['x'], output_names=['y'], dynamic_axes={'y': [0]},
                      opset_version=11, verbose=False)

    onnx_session = onnxruntime.InferenceSession(onnx_path)
    output = onnx_session.run(['y'], {'x': x_array})
    print(output)
