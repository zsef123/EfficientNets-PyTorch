import os
import argparse

import numpy as np

import torch

import tensorflow as tf

import sys
sys.path.append("..")
from models.effnet import EfficientNet

sys.path.append("tf_repo")
from eval_ckpt_main import EvalCkptDriver


def convert_conv(m, weight, bias=None):
    m.weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
    if bias is not None:
        m.bias.data = torch.from_numpy(bias)


def convert_depthwise_conv(m, weight, bias=None):
    m.weight.data = torch.from_numpy(np.transpose(weight, (2, 3, 0, 1)))
    if bias is not None:
        m.bias.data = torch.from_numpy(bias)


def convert_linear(m, weight, bias=None):
    m.weight.data = torch.from_numpy(np.transpose(weight))
    if bias is not None:
        m.bias.data = torch.from_numpy(bias)


def convert_bn(m, gamma, beta, mean, var):
    m.weight.data = torch.from_numpy(gamma)
    m.bias.data = torch.from_numpy(beta)
    m.running_mean.data = torch.from_numpy(mean)
    m.running_var.data = torch.from_numpy(var)


def convert_stem(stem, tf_params):
    convert_conv(stem[0], tf_params[0])
    convert_bn(stem[1], *tf_params[1:])


def convert_head(head, tf_params):
    convert_conv(head[0], tf_params[0])
    convert_bn(head[1], *tf_params[1:5])
    convert_linear(head[-1], *tf_params[5:])


def convert_se(m, tf_params):
    convert_conv(m.se[1], tf_params[0], tf_params[1])
    convert_conv(m.se[3], tf_params[2], tf_params[3])


def convert_MBConv(m, tf_params):
    if isinstance(m.expand_conv, torch.nn.Identity):
        tf_params = [None] * 5 + tf_params
    else:
        convert_conv(m.expand_conv[0], tf_params[0])
        convert_bn(m.expand_conv[1], *tf_params[1:5])

    convert_depthwise_conv(m.depth_wise_conv[0], tf_params[5])
    convert_bn(m.depth_wise_conv[1], *tf_params[6:10])
    convert_se(m.se, tf_params[10:14])
    convert_conv(m.project_conv[0], tf_params[14])
    convert_bn(m.project_conv[1], *tf_params[15:])


def arg_parse():
    desc = "TF EfficientNet to Pytorch EfficientNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='efficientnet-b0')
    parser.add_argument('--tf_weight', type=str, required=True,
                        help='Directory name to save the TF chekpoint')
    parser.add_argument('--pth_weight', type=str, default='model',
                        help='output PyTorch model file name')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    # Loading TF Network
    driver = EvalCkptDriver(args.model)
    image_files = ["dummy"]
    with tf.Graph().as_default(), tf.Session() as sess:
        images, _ = driver.build_dataset(image_files, [0] * len(image_files), False)
        _ = driver.build_model(images, is_training=False)
        tf_keys = [k.name for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

    def key_to_param(ks):
        return [tf.train.load_variable(args.tf_weight, k) for k in ks]

    model = EfficientNet(1, 1)

    tf_stem = [k for k in tf_keys if "stem" in k]
    convert_stem(model.stem, key_to_param(tf_stem))

    tf_head = [k for k in tf_keys if "head" in k]
    convert_head(model.head, key_to_param(tf_head))

    blocks = list(set([k for k in tf_keys if "block" in k]))
    tf_block = [list() for _ in blocks]
    for k in tf_keys:
        if "block" in k:
            # efficientnet-b0/blocks_5/...
            block_id = int(k.split("/")[1].split("_")[1])
            tf_block[block_id].append(k)

    # Flatten All MBConv in MBBlock
    mbconvs = [mbconv for mbblock in model.blocks for mbconv in mbblock.layers]
    for i, (mbconv, tf_mbconv) in enumerate(zip(mbconvs, tf_block)):
        convert_MBConv(mbconv, key_to_param(tf_mbconv))

    torch.save(model.state_dict(), args.pth_weight + ".pth")

    print("\n\nTF to Pytorch Conversion Done")
