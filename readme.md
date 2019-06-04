# EfficientNet

A PyTorch implementation of `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.`


### [[arxiv]](https://arxiv.org/abs/1905.11946) [[Official TF Repo]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)


<hr>

Implement based on Official TF Repo. Only opened EfficientNet is included. <br>
This repo not contains baseline network search(Mnas-Net) and compound coefficient search methods.<br>

<b>Some details(HyperParams, transform, EMA ...) are different with Original repo.</b>


## Pretrained network

This is not end-to-end imagenet trainning weight.
Using the Official TF Pretrained weight.

Please check the [conversion/readme.md](conversion/readme.md) and [pretrained_example.ipynb](pretrained_example.ipynb)


## How to use:

```
python3 main.py -h
usage: main.py [-h] --save_dir SAVE_DIR [--root ROOT] [--gpus GPUS]
               [--num_workers NUM_WORKERS] [--model {b0}] [--epoch EPOCH]
               [--batch_size BATCH_SIZE] [--test]
               [--dropout_rate DROPOUT_RATE]
               [--dropconnect_rate DROPCONNECT_RATE] [--optim {adam,rmsprop}]
               [--lr LR] [--beta [BETA [BETA ...]]] [--momentum MOMENTUM]
               [--eps EPS] [--decay DECAY]

Pytorch EfficientNet

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory name to save the model
  --root ROOT           The Directory of data path.
  --gpus GPUS           Select GPU Numbers | 0,1,2,3 |
  --num_workers NUM_WORKERS
                        Select CPU Number workers
  --model {b0}          The type of Efficient net.
  --epoch EPOCH         The number of epochs
  --batch_size BATCH_SIZE
                        The size of batch
  --test                Only Test
  --dropout_rate DROPOUT_RATE
  --dropconnect_rate DROPCONNECT_RATE
  --optim {adam,rmsprop}
  --lr LR               Base learning rate when train batch size is 256.
  --beta [BETA [BETA ...]]
  --momentum MOMENTUM
  --eps EPS
  --decay DECAY
```

<hr>

### TODO

 - Hyper Parameter / Imagenet Transformation Check
 - Implementation of Resolution Change
 - Validation on Imagenet Dataset
 - Clean up logging

<hr>
