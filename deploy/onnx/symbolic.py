import os
import warnings

import numpy as np
import torch
from torch.nn.modules.utils import _pair, _single, _triple
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    return g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)

def register_extra_symbolics(opset=11):
    # Following strings of text style are from colorama package
    register_op('grid_sampler', grid_sampler, '', opset)
