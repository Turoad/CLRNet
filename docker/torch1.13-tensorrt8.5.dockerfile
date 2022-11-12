FROM nvcr.io/nvidia/pytorch:22.10-py3


# Install deps for CLRNet
RUN pip install pandas \
      addict \
      sklearn \
      opencv-python \
      pytorch_warmup \
      scikit-image \
      tqdm \
      p_tqdm \
      imgaug>=0.4.0 \
      Shapely==1.7.0 \
      ujson==1.35 \
      yapf \
      albumentations==0.4.6 \
      mmcv==1.3.0 \
      pathspec \
      -i https://mirrors.aliyun.com/pypi/simple/
