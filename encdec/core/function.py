# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from encdec.utils.utils import AverageMeter
from encdec.utils.utils import get_confusion_matrix
from encdec.utils.utils import adjust_learning_rate

import encdec.utils.distributed as dist
from tifffile import imsave, imread
from torch.nn.functional import mse_loss, l1_loss


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
    return ave_loss.average()


def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.cuda()

            losses, _ = model(image, label)
            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average()


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=True, evaluate=True):
    model.eval()
    ave_loss = AverageMeter()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()

            # pred = test_dataset.simple_inference(config, model, image)

            pred = model(image)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if evaluate:
                pred = pred.cuda()
                label = label.cuda()
                cur_mse = mse_loss(pred, label).item()
                if index % 100 == 0:
                    logging.info('MSE LOSS at INDEX {}: {}'.format(index, cur_mse))
                ave_loss.update(cur_mse)
            if sv_pred:
                pred = pred.detach().cpu().numpy()[0]
                output_file = os.path.join(sv_dir, name[0])# + test_dataset.lbl_ext)
                if index % 100 == 0:
                    logging.info('Saving prediction {} with shape: {} at {} type: {}'.format(index, pred.shape, output_file, pred.dtype))
                imsave(output_file, pred)

    return ave_loss.average() if evaluate else None


def testval_v2(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=True, evaluate=True):
    model.eval()
    l1 = 0
    l2 = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()

            # pred = test_dataset.simple_inference(config, model, image)
            pred = model(image)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if evaluate:
                pred = pred.cuda()
                label = label.cuda()
                cur_l2 = mse_loss(pred, label).item()
                cur_l1 = l1_loss(pred, label).item()
                l1 += cur_l1
                l2 += cur_l2
                if index % 100 == 0:
                    logging.info('INDEX: {}, L1: {}, L2: {}'.format(index, cur_l1, cur_l2))

            # if sv_pred:
            #     pred = pred.detach().cpu().numpy()[0]
            #     output_file = os.path.join(sv_dir, name[0] + test_dataset.lbl_ext)
            #     if index % 100 == 0:
            #         logging.info('Saving prediction {} with shape: {} at {} type: {}'.format(index, pred.shape, output_file, pred.dtype))
            #     imsave(output_file, pred)

    l1 = l1 / len(test_dataset)
    l2 = l2 / len(test_dataset)
    return l1, l2 if evaluate else None



