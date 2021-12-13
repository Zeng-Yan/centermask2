# nohup python run_shell.py
import torch
import os
import sys
import numpy as np
sys.path.append('./centermask2')


# def abs_err(m1: np.array, m2: np.array):
#     err = np.abs(m1 - m2)
#     print(np.max(err))
#     print(np.sum(err) / err.size)
#     return err

def cos_sim(t1: torch.tensor, t2: torch.tensor) -> torch.tensor:
    if t1.numel() != t2.numel():
        print(f'[WARNING] Shape of tensor1({t1.shape}) not match tensor2({t2.shape}), clip activated')
    l = len(t1) if len(t1) < len(t2) else len(t2)
    t1, t2 = t1[0:l], t2[0:l]
    print(f'Shape of tensor1({t1.shape}) & tensor2({t2.shape})')
    print(f'Dtype of tensor1({t1.dtype}) & tensor2({t2.dtype})')
    sim = torch.cosine_similarity(t1.to(torch.float), t2.to(torch.float), dim=-1)
    print(f'cosine_similarity:\n{sim}')
    print(f'sum: {torch.sum(sim)}')


def mae(m1: torch.tensor, m2: torch.tensor):
    print(f'[WARNING] Shape of tensor1({m1.shape}) not match tensor2({m2.shape}), clip activated')
    l = len(m1) if len(m1) < len(m2) else len(m2)
    m1, m2 = m1[::l], m2[::l]
    err = torch.abs(m1 - m2)
    print(torch.max(err))
    print(f'MAE of elements:{torch.sum(err) / err.numel()}')
    return err


def buf_to_tensor(b: np.array, shape: tuple) -> torch.tensor:
    return torch.from_numpy(b).reshape(shape)


if __name__ == '__main__':
    shape = (-1, 2)
    path_bin_pth = 'candidate.bin'
    path_bin_om = "result/dumpOutput_device0/000000000139_1.bin"
    out_tensor = '1938'
    out_node = 'Gather_955'
    out_shape = '1000, 2'
    out_type = 'float'

    if os.path.exists(path_bin_om):
        os.remove(path_bin_om)

    # mod onnx
    cmd = f'python check_layers_outputs.py --config-file "centermask2/configs/centermask/zy_model_config.yaml" \
        --module Cast_1687 --node {out_node} --tensor {out_tensor} --type {out_type} --shape {out_shape} \
        MODEL.WEIGHTS "/home/zeng/centermask2-V-39-eSE-FPN-ms-3x.pth" MODEL.DEVICE cpu'
    # os.system(cmd)
    os.system('bash to_om.sh > sub_om.log')

    # onnx to om
    # os.system('bash env.sh')
    # cmd = 'atc --framework=5 --model=sub.onnx --output=centermask_sub \
    #     --input_format=NCHW --input_shape="img:1,3,1344,1344" --log=debug --soc_version=Ascend310'
    # os.system(cmd + ' > onnx2om.log')

    ####


    # om inference
    cmd = './benchmark.x86_64 -model_type=vision -om_path=centermask_sub.om -device_id=0 -batch_size=1 ' \
          '-input_text_path=139_bin.info -input_width=1344 -input_height=1344 -useDvpp=false ' \
          '-output_binary=true'
    stats = os.system(cmd)

    # read results
    buf = np.fromfile(path_bin_om, dtype='int64')
    buf = buf_to_tensor(buf, shape)
    print('\n' * 10)
    # print(buf.shape, buf.dtype)
    print(buf)

    cmp_r = np.fromfile(path_bin_pth, dtype='int64')
    cmp_r = buf_to_tensor(cmp_r, shape)

    # print(cmp_r.shape, )
    print(cmp_r)

    cos_sim(buf, cmp_r)
    mae(buf, cmp_r)


