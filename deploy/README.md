
# Deployment for CLRNet

We provide scripts for ONNX and TensorRT deployment. The recommended onnx opset version is higher than 16, and TensorRT version is higher than 8.5.

## Export from Pytorch to ONNX 

```
python deploy/onnx/export_onnx.py --help
usage: export_onnx.py [-h] [--cfg CFG] [--out-file OUT_FILE] [--load-from LOAD_FROM] [--opset-version OPSET_VERSION]

Convert CLRNet models from pytorch to ONNX

optional arguments:
  -h, --help            show this help message and exit
  --cfg CFG             Filename of input ONNX model
  --out-file OUT_FILE   Filename of output onnx
  --load-from LOAD_FROM
                        Filename of pretrained model path
  --opset-version OPSET_VERSION
                        Opset verison of onnx
```

For example:
```shell
python deploy/onnx/export_onnx.py --cfg configs/clrnet/clr_resnet18_culane.py \
                                  --load-from culane_r18.pth \
                                  --opset-version 16 \
                                  --out-file culane_r18.onnx
```

if meet error, you can try to use lower opset-version, eg: when you use the provided trt7.2 version docker, you should set opset-version <= 12. 

## Export from ONNX to TensorRT

```
python deploy/tensorrt/onnx2trt.py --help
usage: onnx2trt.py [-h] [--model MODEL] [--trt_file TRT_FILE] [--input_img INPUT_IMG] [--show] [--shape SHAPE [SHAPE ...]] [--min-shape MIN_SHAPE [MIN_SHAPE ...]] [--max-shape MAX_SHAPE [MAX_SHAPE ...]] [--workspace-size WORKSPACE_SIZE] [--verbose] config

Convert CLRNet models from ONNX to TensorRT

positional arguments:
  config                config file path

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Filename of input ONNX model
  --trt_file TRT_FILE   Filename of output TensorRT engine
  --input_img INPUT_IMG
                        Image for test
  --show                Whether to show detection results
  --shape SHAPE [SHAPE ...]
                        Input size of the model
  --min-shape MIN_SHAPE [MIN_SHAPE ...]
                        Minimum input size of the model in TensorRT
  --max-shape MAX_SHAPE [MAX_SHAPE ...]
                        Maximum input size of the model in TensorRT
  --workspace-size WORKSPACE_SIZE
                        Max workspace size in GiB
  --verbose             Whether to verbose logging messages while creating TensorRT engine. Defaults to False.
```

For example:
```
# This will generate tensorrt engine name `culane_r18.trt`.
python deploy/tensorrt/onnx2trt.py configs/clrnet/clr_resnet18_culane.py \
                                  --model culane_r18.onnx \
                                  --trt_file culane_r18.trt

# This will show the detection result of the tensorrt engine in tmp.png
python deploy/tensorrt/onnx2trt.py configs/clrnet/clr_resnet18_culane.py \
                                  --trt_file culane_r18.trt \
                                  --show \
                                  --input_img data/CULane/driver_23_30frame/05160756_0456.MP4/00075.jpg
```

## Lower version of ONNX or TensorRT
CLRNet uses operators which are not supported with previous versions of ONNX and TensorRT, so we need to build the custom operator for these versions.


### Install TensorRT
Download the corresponding TensorRT build from [NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-tensorrt-download).

For example, for Ubuntu 16.04 on x86-64 with cuda-10.2, the downloaded file is `TensorRT-7.2.1.6.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz`.

```
pip install $TENSORRT_DIR/python/tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
```

### Install onnxruntime
```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1 
```

### Build the operator for ONNX and TensorRT
```
cd /path of CLRNet/deploy/csrc/
mkdir build && cd build
cmake ..
make
```


## Docker Support
We recommend you to use our established PyTorch docker image:
```
docker pull turoad/clrnet:torch1.13-tensorrt8.5
```
This docker is based on the base image `nvcr.io/nvidia/pytorch:22.10-py3`, you can find the details in [pytorch-release-notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-10.html#rel-22-10). Another pre-build docker is `turoad/clrnet:torch1.8-tensorrt7.2` which is base on ` nvcr.io/nvidia/pytorch:21.02-py3`.


Finally, follows the [README](https://github.com/Turoad/CLRNet/blob/main/README.md#use-docker-to-run-clrnet-recommended) to run the deployment.

For `turoad/clrnet:torch1.8-tensorrt7.2`, you still need to build the `deploy/csrc` following [Build the operator for ONNX and TensorRT](#build-the-operator-for-onnx-and-tensorrt).

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdeploy](https://github.com/open-mmlab/mmdeploy)
<!--te-->