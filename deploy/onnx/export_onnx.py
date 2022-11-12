import torch
import sys
import os
from clrnet.models.nets.detector import Detector
from clrnet.utils import Config
import argparse
from clrnet.utils.net_utils import load_network


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CLRNet models from pytorch to ONNX')
    parser.add_argument('--cfg',
                        type=str,
                        default='configs/clrnet/clr_resnet18_culane.py',
                        help='Filename of input ONNX model')
    parser.add_argument('--out-file',
                        type=str,
                        default='tmp.onnx',
                        help='Filename of output onnx')

    parser.add_argument('--load-from',
                        type=str,
                        default=None,
                        help='Filename of pretrained model path')

    parser.add_argument('--opset-version',
                        type=int,
                        default=16,
                        help='Opset verison of onnx')

    args = parser.parse_args()
    return args


def parse(args):
    opset_version = args.opset_version
    if opset_version < 16:
        from deploy.onnx.symbolic import register_extra_symbolics
        register_extra_symbolics(opset_version)
    cfg = Config.fromfile(args.cfg)
    cfg.load_from = args.load_from
    net = Detector(cfg)
    net.eval()
    load_network(net, cfg.load_from, logger=None, remove_module_prefix=True)

    x = torch.randn(1, 3, cfg.img_h, cfg.img_w)

    torch.onnx.export(net,
                      args=(x, ),
                      f=args.out_file,
                      input_names=['input'],
                      output_names=['pred'],
                      verbose=False,
                      opset_version=opset_version)


def main():
    args = parse_args()
    parse(args)


if __name__ == '__main__':
    main()
