# Pretrained TF to Pytorch

[Official TF Repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
[Official TF Repo Snapshot](https://github.com/mingxingtan/efficientnet)

## Usage

1. Download TF checkpoint

```
export MODEL=efficientnet-b0
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/${MODEL}.tar.gz
tar zxf ${MODEL}.tar.gz
```

2. Run `convert.py`

```
python3 convert.py -h
usage: convert.py [-h] [--model MODEL] --tf_weight TF_WEIGHT
                  [--pth_weight PTH_WEIGHT]

TF EfficientNet to Pytorch EfficientNet

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL
  --tf_weight TF_WEIGHT
                        Directory name to save the TF chekpoint
  --pth_weight PTH_WEIGHT
                        output PyTorch model file name
```

Example
```
python3 convert.py --model efficientnet-b0 --tf efficientnet-b0 --pth b0
```