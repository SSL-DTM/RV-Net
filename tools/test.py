# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import pandas as pd



from encdec import models
from encdec.config import config
from encdec import data
from encdec.config import update_config
from torch.nn import MSELoss
from encdec.utils.modelsummary import get_model_summary
from encdec.utils.utils import create_logger, FullModel
from encdec.core.function import train, validate, testval
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.' + config.MODEL.NAME +
                 '.get_model')(config)

    if config.MODEL.NAME != 'swin':
        dump_input = torch.rand(
            (1, config.DATASET.INPUT_CHANNELS, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )
        logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    test_dataset = eval('data.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        split=config.DATASET.TEST_DIR,
        img_dirs=config.DATASET.IMG_DIRS,
        lbl_dirs=config.DATASET.LBL_DIRS,
        num_output_channels=config.DATASET.NUM_OUTPUT_CHANNELS,
        multi_scale=False,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        img_ext=config.DATASET.IMG_EXT,
        lbl_ext=config.DATASET.LBL_EXT)

    logger.info('len test data: {}'.format(len(test_dataset)))

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)


    start = timeit.default_timer()
    prediction_folder = os.path.basename(args.cfg).split('.')[0]
    sv_dir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_DIR, prediction_folder) if not \
        config.TEST.JUST_PREDICT else os.path.join(os.path.split(config.DATASET.ROOT)[0], 'tmp', prediction_folder)
    logger.info('Prediction folder: {}'.format(sv_dir))
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir, exist_ok=False)

    eval_flag = False if config.TEST.JUST_PREDICT else True
    result = testval(config, test_dataset, testloader, model, sv_dir=sv_dir, sv_pred=True, evaluate=eval_flag)
    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    if result:
        logger.info('MSE: {}'.format(result))
    logger.info('Done')


if __name__ == '__main__':
    main()