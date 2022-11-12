import os
import os.path as osp
import argparse
import warnings
import numpy as np
import onnx
import torch
from mmcv import Config
import tensorrt as trt
import cv2
from deploy.tensorrt.tensorrt_utils import onnx2trt, save_trt_engine, TRTWraper

if trt.__version__ < '8.5':
    from deploy.tensorrt.tensorrt_utils import load_tensorrt_plugin
    load_tensorrt_plugin()

from clrnet.datasets.process import Process
from clrnet.utils.config import Config
from clrnet.models.registry import build_net
from clrnet.utils.visualization import imshow_lanes


def get_GiB(x: int):
    """return x GiB."""
    return x * (1 << 30)


def test_speed(data, trt_file):
    from mmcv.tensorrt import TRTWraper, load_tensorrt_plugin
    try:
        load_tensorrt_plugin()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with TensorRT from source.')

    output_names = ['pred']
    model = TRTWraper(trt_file, ['input'], output_names)

    # imgs = [torch.randn(1, 3, 320, 800).cuda()]
    # input_data = imgs[0].contiguous()
    input_data = data['img'].cuda()
    with torch.cuda.device(0), torch.no_grad():
        tot = 0
        import time
        for i in range(1000):
            outputs = model({'input': input_data})

        for i in range(1000):
            t0 = time.time()
            outputs = model({'input': input_data})
            t1 = time.time()
            outputs = [outputs[name] for name in model.output_names]
            tot += t1 - t0
        print(tot)
    return outputs


def load_trt_model(trt_file, input_names=['input'], output_names=['pred']):
    model = TRTWraper(trt_file, input_names, output_names)
    return model


def inference(input_data, model):
    outputs = model({'input': input_data})
    outputs = [outputs[name] for name in model.output_names]
    return outputs


def preprocess(cfg, img_path):
    processes = Process(cfg.val_process, cfg)
    ori_img = cv2.imread(img_path)
    img = ori_img[cfg.cut_height:, :, :].astype(np.float32)
    data = {'img': img, 'lanes': []}
    data = processes(data)
    data['img'] = data['img'].unsqueeze(0)
    data.update({'img_path': img_path, 'ori_img': ori_img})
    return data


def show_result(cfg_path, input_image, trt_model, out_file='./tmp.png'):
    cfg = Config.fromfile(cfg_path)
    net = build_net(cfg)
    data = preprocess(cfg, input_image)
    output = inference(data['img'].cuda(), model)
    lanes = net.get_lanes(output[0])[0]
    lanes = [lane.to_array(cfg) for lane in lanes]
    imshow_lanes(data['ori_img'], lanes, out_file=out_file)


def onnx2tensorrt(cfg, onnx_file,
                  trt_file,
                  input_config,
                  verbose=False,
                  show=False,
                  workspace_size=1):
    onnx_model = onnx.load(onnx_file)
    max_shape = input_config['max_shape']
    min_shape = input_config['min_shape']
    opt_shape = input_config['opt_shape']
    fp16_mode = False
    # create trt engine and wrapper
    opt_shape_dict = {'input': [min_shape, opt_shape, max_shape]}
    max_workspace_size = get_GiB(workspace_size)
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        log_level=trt.Logger.VERBOSE if verbose else trt.Logger.ERROR,
        fp16_mode=fp16_mode,
        max_workspace_size=max_workspace_size)
    save_dir, _ = osp.split(trt_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_trt_engine(trt_engine, trt_file)
    print(f'Successfully created TensorRT engine: {trt_file}')

    if show:
        show_result(cfg, input_config['input_path'], trt_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CLRNet models from ONNX to TensorRT')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--model',
                        type=str,
                        default='tmp.onnx',
                        help='Filename of input ONNX model')
    parser.add_argument('--trt_file',
                        type=str,
                        default='tmp.trt',
                        help='Filename of output TensorRT engine')
    parser.add_argument('--input_img',
                        type=str,
                        default='',
                        help='Image for test')
    parser.add_argument('--show',
                        action='store_true',
                        help='Whether to show output results')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[320, 800],
                        help='Input size of the model')
    parser.add_argument('--min-shape',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Minimum input size of the model in TensorRT')
    parser.add_argument('--max-shape',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Maximum input size of the model in TensorRT')
    parser.add_argument('--workspace-size',
                        type=int,
                        default=1,
                        help='Max workspace size in GiB')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to verbose logging messages while creating \
                TensorRT engine. Defaults to False.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    def parse_shape(shape):
        if len(shape) == 1:
            shape = (1, 3, shape[0], shape[0])
        elif len(args.shape) == 2:
            shape = (1, 3) + tuple(shape)
        else:
            raise ValueError('invalid input shape')
        return shape

    if args.shape:
        input_shape = parse_shape(args.shape)

    if not args.max_shape:
        max_shape = input_shape
    else:
        max_shape = parse_shape(args.max_shape)

    if not args.min_shape:
        min_shape = input_shape
    else:
        min_shape = parse_shape(args.min_shape)

    input_config = {
        'min_shape': min_shape,
        'opt_shape': input_shape,
        'max_shape': max_shape,
        'input_shape': input_shape,
        'input_path': args.input_img,
    }
    if args.show:
        model = load_trt_model(args.trt_file)
        show_result(args.config, args.input_img, model)
    else:
        # Create TensorRT engine
        onnx2tensorrt(args.config, args.model,
                      args.trt_file,
                      input_config,
                      show=args.show,
                      workspace_size=args.workspace_size,
                      verbose=args.verbose)
