'''
Description:
Author: weihuang
Date: 2021-11-26 09:22:42
LastEditors: Please set LastEditors
LastEditTime: 2021-11-29 20:14:18
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.utils import setup_seed
from dataset.target_dataset_mito import targetDataSet, targetDataSet_val_twoimgs, Evaluation
from model.CoDetectionCNN import CoDetectionCNN
from model.discriminator_damtnet import labelDiscriminator, featureDiscriminator
from model.discriminator_davsn import get_fc_discriminator
from loss.loss import CrossEntropy2d, BCELoss
from utils.metrics import dice_coeff
from utils.show import show_training, save_prediction_image, show_training2
from utils.utils import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.utils import inference_results
from utils.utils import get_current_consistency_weight

import warnings

warnings.filterwarnings("ignore")


def init_project(cfg):
    print('Initialization ... ', end='', flush=True)
    t1 = time.time()

    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.TRAIN.log_path, model_name)
    cfg.valid_path = os.path.join(cfg.TRAIN.valid_path, model_name)
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    print('Done (time: %.2fs)' % (time.time() - t1))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    source_data = sourceDataSet(cfg.DATA.data_dir_img,
                                cfg.DATA.data_dir_label,
                                cfg.DATA.data_list,
                                # target_path=cfg.DATA.data_dir_target,
                                crop_size=(cfg.DATA.input_size, cfg.DATA.input_size),
                                stride=cfg.DATA.source_stride,
                                strong_aug=cfg.DATA.strong_aug_source)
    train_provider = torch.utils.data.DataLoader(source_data,
                                                 batch_size=cfg.TRAIN.batch_size,
                                                 shuffle=True,
                                                 num_workers=cfg.TRAIN.num_workers)
    if cfg.TRAIN.if_valid:
        val_data = targetDataSet_val_twoimgs(cfg.DATA.data_dir_val,
                                             cfg.DATA.data_dir_val_label,
                                             cfg.DATA.data_list_val,
                                             crop_size=(cfg.DATA.input_size_test, cfg.DATA.input_size_test),
                                             stride=cfg.DATA.target_stride)
        valid_provider = torch.utils.data.DataLoader(val_data,
                                                     batch_size=1,
                                                     shuffle=False)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                           n_classes=cfg.MODEL.output_nc).to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


# def calculate_lr(iters):
#     if iters < cfg.TRAIN.warmup_iters:
#         current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
#     else:
#         if iters < cfg.TRAIN.decay_iters:
#             current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
#         else:
#             current_lr = cfg.TRAIN.end_lr
#     return current_lr

def cross_supervision(cpred, apred, dpred, criterion_seg):
    cpred_detach = cpred.clone().detach()
    cpred_detach = torch.argmax(cpred_detach, dim=1)
    apred_detach = apred.clone().detach()
    apred_detach = torch.argmax(apred_detach, dim=1)
    dpred_detach = dpred.clone().detach()
    dpred_detach = torch.argmax(dpred_detach, dim=1)
    clabel_cross = torch.abs(apred_detach - dpred_detach)
    alabel_cross = torch.abs(cpred_detach - dpred_detach)
    loss_cpred_cross = criterion_seg(cpred, clabel_cross.long())
    loss_apred_cross = criterion_seg(apred, alabel_cross.long())
    return loss_cpred_cross, loss_apred_cross

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def feature_alignment(fea_src, fea_tgt, gt_src, threshold=0.7):
    var_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)
    align_loss = torch.tensor(0, dtype=fea_src.dtype, device=fea_src.device)

    fea_src = F.normalize(fea_src, p=2, dim=1)  # L2 norm
    fea_tgt = F.normalize(fea_tgt, p=2, dim=1)  # L2 norm
    feature_dim = fea_src.shape[1]

    assert fea_src.shape[0] == gt_src.shape[0]

    # discriminative loss used for source domain
    num_classes = 2 # two classes
    proto = []
    fea_src = fea_src.permute(0, 2, 3, 1)
    for k in range(num_classes):
        mask_k = gt_src == k
        emb_k = fea_src[mask_k, :]
        num = emb_k.shape[0]
        if num == 0: continue

        proto_k = l2_normalize(torch.mean(emb_k, dim=0))
        proto.append(proto_k)

        var_loss += (1 - torch.mm(emb_k, proto_k.reshape(feature_dim, 1))).pow(2).mean() / num_classes

    proto = torch.stack(proto)
    if proto.shape[0] == 2:
        dist_loss = torch.sum(proto[0] * proto[1]).pow(2)

        # alignment
        b, c, h, w = fea_tgt.shape
        fea_tgt_emb = fea_tgt.permute(0, 2, 3, 1)
        fea_tgt_emb = torch.reshape(fea_tgt_emb, (-1, feature_dim))
        fea_tgt_emb = l2_normalize(fea_tgt_emb)
        proto = l2_normalize(proto)
        seg_tgt = torch.einsum('nd,md->nm', fea_tgt_emb, proto)

        for k in range(proto.shape[0]):
            seg_tgt_k = seg_tgt[:, k]
            mask_k = seg_tgt_k > threshold
            emb_k = seg_tgt_k[mask_k]
            num = emb_k.shape[0]
            if num == 0: continue

            align_loss += (1 - emb_k.pow(2).mean()) / num_classes

        seg_tgt = torch.reshape(seg_tgt, (b, h, w, -1))
        seg_tgt = seg_tgt.permute(0, 3, 1, 2)
    else:
        dist_loss = 0.0
        align_loss = 0.0
        seg_tgt = None

    return var_loss, dist_loss, align_loss, seg_tgt


def loop(cfg, train_provider, valid_provider, model, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    # f_loss_adv_txt = open(os.path.join(cfg.record_path, 'loss_adv.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0.0
    sum_loss_total = 0.0
    sum_loss_supervised = 0.0
    sum_loss_fa = 0.0
    sum_loss_proto_pred = 0.0
    sum_loss_consist_fea = 0.0
    sum_loss_consist_pl = 0.0
    sum_loss_warp_target = 0.0
    target_stride = cfg.DATA.target_stride
    device = torch.device('cuda:0')

    # build dataloder for target dataset
    target_data = targetDataSet(cfg.DATA.data_dir_target,
                                cfg.DATA.data_dir_target_label,
                                cfg.DATA.data_list_target,
                                crop_size=(cfg.DATA.input_size_target, cfg.DATA.input_size_target),
                                stride=cfg.DATA.target_stride,
                                strong_aug=cfg.DATA.strong_aug_target)
    targetloader = torch.utils.data.DataLoader(target_data,
                                               batch_size=cfg.TRAIN.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.TRAIN.num_workers)

    target_evaluation = Evaluation(root_label=cfg.DATA.data_dir_val_label,
                                   crop_size=(cfg.DATA.input_size_test, cfg.DATA.input_size_test))
    trainloader_iter = enumerate(train_provider)
    targetloader_iter = enumerate(targetloader)
    # if_adv_weight = cfg.TRAIN.if_adv_weight
    #
    # model_spatial = get_fc_discriminator(num_classes=cfg.MODEL.num_classes).to(device)
    # model_temporal = get_fc_discriminator(num_classes=cfg.MODEL.num_classes).to(device)
    # model_spatial.train()
    # model_temporal.train()
    #
    # # build optimizer for discriminator
    # optimizer_model_spatial = optim.Adam(model_spatial.parameters(), lr=cfg.TRAIN.learning_rate_ms,
    #                                      betas=(0.9, 0.99))
    # optimizer_model_temporal = optim.Adam(model_temporal.parameters(), lr=cfg.TRAIN.learning_rate_mt,
    #                                       betas=(0.9, 0.99))

    # build loss functions
    criterion_seg = CrossEntropy2d().to(device)
    criterion_con = nn.MSELoss()  # consistency loss

    while iters <= cfg.TRAIN.total_iters:
        iters += 1
        t1 = time.time()
        model.train()

        optimizer.zero_grad()
        # optimizer_model_spatial.zero_grad()
        # optimizer_model_temporal.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, iters, cfg.TRAIN.learning_rate, cfg.TRAIN.total_iters, cfg.TRAIN.power)
        # adjust_learning_rate_discriminator(optimizer_model_spatial, iters, cfg.TRAIN.learning_rate_ms,
        #                                    cfg.TRAIN.total_iters, cfg.TRAIN.power)
        # adjust_learning_rate_discriminator(optimizer_model_temporal, iters, cfg.TRAIN.learning_rate_mt,
        #                                    cfg.TRAIN.total_iters, cfg.TRAIN.power)
        #
        # # train G
        # for param in model_spatial.parameters():
        #     param.requires_grad = False
        #
        # for param in model_temporal.parameters():
        #     param.requires_grad = False

        # train with source
        _, batch = trainloader_iter.__next__()
        cimg_source, clabel_source, aimg_source, alabel_source, dlabel_source = batch
        cimg_source = cimg_source.to(device)
        aimg_source = aimg_source.to(device)
        clabel_source = clabel_source.to(device)
        alabel_source = alabel_source.to(device)
        dlabel_source = dlabel_source.to(device)

        _, batch = targetloader_iter.__next__()
        cimg_target, cimg_target_aug, aimg_target, aimg_target_aug, _ = batch
        cimg_target = cimg_target.to(device)
        aimg_target = aimg_target.to(device)
        cimg_target_aug = cimg_target_aug.to(device)
        aimg_target_aug = aimg_target_aug.to(device)

        img_source_cat = torch.cat([cimg_source, aimg_source], dim=1)
        img_tatget_aug_cat = torch.cat([cimg_target_aug, aimg_target_aug], dim=1)
        cpred_source, apred_source, dpred_source, fea_source_c, fea_source_a, fea_source_d = model(img_source_cat)
        cpred_target_aug, apred_target_aug, dpred_target_aug, fea_tatget_aug_c, fea_tatget_aug_a, fea_tatget_aug_d = model(img_tatget_aug_cat)

        loss_cpred = criterion_seg(cpred_source, clabel_source.long())
        loss_apred = criterion_seg(apred_source, alabel_source.long())
        loss_diff = criterion_seg(dpred_source, dlabel_source.long())

        loss_total = loss_cpred + loss_apred + loss_diff
        sum_loss_supervised += loss_total.item()

        # train with target
        img_target_cat = torch.cat([cimg_target, aimg_target], dim=1)
        with torch.no_grad():
            cpred_target, apred_target, dpred_target, fea_tatget_c, fea_tatget_a, fea_tatget_d = model(img_target_cat)
            cpred_target_pl = torch.argmax(cpred_target, dim=1).long()
            apred_target_pl = torch.argmax(apred_target, dim=1).long()
            dpred_target_pl = torch.argmax(dpred_target, dim=1).long()

        if cfg.TRAIN.consistency_weight_rampup:
            weight_fa = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_fa, consistency_rampup=cfg.TRAIN.rampup_iters)
            weight_proto_pred = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_proto_pred, consistency_rampup=cfg.TRAIN.rampup_iters)
            weight_consist_fea = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_consist_fea, consistency_rampup=cfg.TRAIN.rampup_iters)
            weight_consist_pl = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_consist_pl, consistency_rampup=cfg.TRAIN.rampup_iters)
            weight_warp = get_current_consistency_weight(iters, consistency=cfg.TRAIN.weight_warp, consistency_rampup=cfg.TRAIN.rampup_iters)
        else:
            weight_fa = cfg.TRAIN.weight_fa
            weight_proto_pred = cfg.TRAIN.weight_proto_pred
            weight_consist_fea = cfg.TRAIN.weight_consist_fea
            weight_consist_pl = cfg.TRAIN.weight_consist_pl
            weight_warp = cfg.TRAIN.weight_warp

        # feature alignment loss
        if cfg.TRAIN.feature_align:
            loss_var_c, loss_dist_c, loss_aglin_c, proto_pred_c = feature_alignment(fea_source_c, fea_tatget_aug_c, clabel_source.long(), threshold=cfg.TRAIN.threshold)
            loss_var_a, loss_dist_a, loss_aglin_a, proto_pred_a = feature_alignment(fea_source_a, fea_tatget_aug_a, alabel_source.long(), threshold=cfg.TRAIN.threshold)
            loss_var_d, loss_dist_d, loss_aglin_d, proto_pred_d = feature_alignment(fea_source_d, fea_tatget_aug_d, dlabel_source.long(), threshold=cfg.TRAIN.threshold)
            loss_fa = (loss_var_c + loss_var_a + loss_var_d) / 3.0 \
                      + (loss_dist_c + loss_dist_a + loss_dist_d) / 3.0 \
                      + (loss_aglin_c + loss_aglin_a + loss_aglin_d) / 3.0

            loss_fa = loss_fa * weight_fa
            loss_total += loss_fa
            sum_loss_fa += loss_fa.item()

        # prototype mask loss
        if cfg.TRAIN.proto_pred_loss:
            if proto_pred_c is None or proto_pred_a is None or proto_pred_d is None:
                sum_loss_proto_pred = 0.0
            else:
                loss_proto_pred = criterion_seg(proto_pred_c, cpred_target_pl) / 3.0 + \
                                criterion_seg(proto_pred_a, apred_target_pl) / 3.0 + \
                                criterion_seg(proto_pred_d, dpred_target_pl) / 3.0
                loss_proto_pred = loss_proto_pred * weight_proto_pred
                loss_total += loss_proto_pred
                sum_loss_proto_pred += loss_proto_pred.item()
        else:
            sum_loss_proto_pred = 0.0

        # consistent feature loss
        if cfg.TRAIN.consist_fea_loss:
            loss_consist_fea = criterion_con(fea_tatget_aug_c, fea_tatget_c) / 3.0 + \
                            criterion_con(fea_tatget_aug_a, fea_tatget_a) / 3.0 + \
                            criterion_con(fea_tatget_aug_d, fea_tatget_d) / 3.0
            loss_consist_fea = loss_consist_fea * weight_consist_fea
            loss_total += loss_consist_fea
            sum_loss_consist_fea += loss_consist_fea.item()
        else:
            sum_loss_consist_fea = 0.0

        # consistent pseudo label loss
        if cfg.TRAIN.consist_pl_loss:
            loss_consist_pl = criterion_seg(cpred_target_aug, cpred_target_pl) / 3.0 + \
                            criterion_seg(apred_target_aug, apred_target_pl) / 3.0 + \
                            criterion_seg(dpred_target_aug, dpred_target_pl) / 3.0
            loss_consist_pl = loss_consist_pl * weight_consist_pl
            loss_total += loss_consist_pl
            sum_loss_consist_pl += loss_consist_pl.item()
        else:
            sum_loss_consist_pl = 0.0

        # warp loss
        if cfg.TRAIN.warp_loss:
            loss_warp_c, loss_warp_a = cross_supervision(cpred_target_aug, apred_target_aug, dpred_target_aug, criterion_seg)
            loss_warp = (loss_warp_c + loss_warp_a) / 2.0
            loss_warp = loss_warp * weight_warp
            loss_total += loss_warp
            sum_loss_warp_target += loss_warp.item()
        else:
            sum_loss_warp_target = 0.0
        loss_total.backward()
        sum_loss_total += loss_total.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        # optimizer_model_spatial.step()
        # optimizer_model_temporal.step()
        learning_rate = optimizer.param_groups[0]['lr']

        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss=%.6f, loss_sup=%.6f, loss_fa=%.6f, '
                             'loss_proto=%.6f, loss_fea=%.6f, loss_pl=%.6f, loss_warp=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss_total, sum_loss_supervised, sum_loss_fa, sum_loss_proto_pred, sum_loss_consist_fea, sum_loss_consist_pl, sum_loss_warp_target, learning_rate, sum_time,
                    (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                # logging.info('step %d, loss=%.6f, loss_cross=%.6f, loss_adv_spatial=%.6f, loss_adv_temporal=%.6f'
                #              % (
                #              iters, sum_loss_adv, sum_loss_cross_target, sum_loss_adv_spatial, sum_loss_adv_temporal))
            else:
                logging.info('step %d, loss=%.6f, loss_sup=%.6f, loss_fa=%.6f, loss_proto=%.6f, loss_fea=%.6f, loss_pl=%.6f, loss_warp=%.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                    % (iters, \
                                sum_loss_total / cfg.TRAIN.display_freq, \
                       sum_loss_supervised / cfg.TRAIN.display_freq, \
                                sum_loss_fa / cfg.TRAIN.display_freq, \
                                sum_loss_proto_pred / cfg.TRAIN.display_freq, \
                                sum_loss_consist_fea / cfg.TRAIN.display_freq, \
                                sum_loss_consist_pl / cfg.TRAIN.display_freq, \
                                sum_loss_warp_target / cfg.TRAIN.display_freq, learning_rate, sum_time, \
                       (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss_sup', sum_loss_supervised / cfg.TRAIN.display_freq, iters)
                f_loss_txt.write('step=%d, loss=%.6f, loss_sup=%.6f, loss_fa=%.6f, loss_proto=%.6f, loss_fea=%.6f, loss_pl=%.6f, loss_warp=%.6f' % \
                    (iters, sum_loss_total / cfg.TRAIN.display_freq, \
                                sum_loss_supervised / cfg.TRAIN.display_freq, \
                                sum_loss_fa / cfg.TRAIN.display_freq, \
                                sum_loss_proto_pred / cfg.TRAIN.display_freq, \
                                sum_loss_consist_fea / cfg.TRAIN.display_freq, \
                                sum_loss_consist_pl / cfg.TRAIN.display_freq, \
                                sum_loss_warp_target / cfg.TRAIN.display_freq))
                f_loss_txt.write('\n')
                f_loss_txt.flush()
                # f_loss_adv_txt.write(
                #     'step=%d, loss=%.6f, loss_cross=%.6f, loss_adv_spatial=%.6f, loss_adv_temporal=%.6f' % \
                #     (iters, sum_loss_adv / cfg.TRAIN.display_freq,
                #      sum_loss_cross_target / cfg.TRAIN.display_freq,
                #      sum_loss_adv_spatial / cfg.TRAIN.display_freq,
                #      sum_loss_adv_temporal / cfg.TRAIN.display_freq))
                # f_loss_adv_txt.write('\n')
                # f_loss_adv_txt.flush()
                sys.stdout.flush()
                sum_time = 0.0
                sum_loss_total = 0.0
                sum_loss_supervised = 0.0
                sum_loss_fa = 0.0
                sum_loss_proto_pred = 0.0
                sum_loss_consist_fea = 0.0
                sum_loss_consist_pl = 0.0
                sum_loss_warp_target = 0.0

        # valid
        if cfg.TRAIN.if_valid:
            if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
                model.eval()
                preds = np.zeros((100, cfg.DATA.input_size_test, cfg.DATA.input_size_test), dtype=np.uint8)
                for i_pic, (cimg, aimg) in enumerate(valid_provider):
                    cimg = cimg.to(device)
                    aimg = aimg.to(device)
                    img_cat = torch.cat([cimg, aimg], dim=1)
                    with torch.no_grad():
                        cpred, apred, _, _, _, _ = model(img_cat)
                    preds[i_pic] = inference_results(cpred, preds[i_pic], mode='mito')
                    preds[i_pic+target_stride] = inference_results(apred, preds[i_pic+target_stride], mode='mito')
                F1 = target_evaluation(preds)
                logging.info('model-%d, F1=%.6f' % (iters, F1))
                writer.add_scalar('valid/F1', F1, iters)
                f_valid_txt.write('model-%d, F1=%.6f' % (iters, F1))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    # f_loss_adv_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)
    cfg.path = cfg_file
    cfg.time = time_stamp

    try:
        if cfg.DATA.aug_chang:
            from dataset.source_dataset_mito import sourceDataSet_chang as sourceDataSet

            print('Import sourceDataSet_chang')
        else:
            from dataset.source_dataset_mito import sourceDataSet
    except:
        from dataset.source_dataset_mito import sourceDataSet

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        if cfg.TRAIN.opt_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.learning_rate, betas=(0.9, 0.99))
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=cfg.TRAIN.learning_rate,
                                  momentum=0.9,
                                  weight_decay=0.0005)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, optimizer, init_iters, writer)
        writer.close()
    else:
        raise NotImplementedError
    print('***Done***')