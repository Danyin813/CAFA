'''
Description: 
Author: weihuang
Date: 2021-11-16 21:17:31
LastEditors: Please set LastEditors
LastEditTime: 2023-01-13 21:14:37
'''

import os
import cv2
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict
from collections import OrderedDict
import h5py
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

from model.CoDetectionCNN import CoDetectionCNN
from dataset.target_dataset import targetDataSet_val_twoimgs, Evaluation
from utils.utils import inference_results, inference_results2
from utils.show import show_test

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='vnc2lucchi1', help='config file')
    parser.add_argument('-mn', '--model_name', type=str, default='vnc2lucchi1')
    parser.add_argument('-mm', '--mode_map', type=str, default='map')
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    trained_model = args.model_name
    out_path = os.path.join('/braindat/lab/yd/august/meta-diffusion/stage2/inference/vnc2lucchi2/mithout_meta_data/', trained_model)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('out_path: ' + out_path)
    seg_img_path = os.path.join(out_path, 'seg_img')
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                           n_classes=cfg.MODEL.output_nc).to(device)

    #ckpt_path = os.path.join('/braindat/lab/yd/august/meta-diffusion/stage2/pretrained/v2l2-model-135000.ckpt')
    ckpt_path = os.path.join('/braindat/lab/yd/august/meta-diffusion/stage2/segnet/model-005000.ckpt')
    checkpoint = torch.load(ckpt_path)
    new_state_dict = OrderedDict()
    # state_dict = checkpoint['model_weights']
    model.load_state_dict(checkpoint['model_weights'])
    # for k, v in state_dict.items():
    #     # name = k[7:] # remove module.
    #     name = k
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    val_data = targetDataSet_val_twoimgs(cfg.DATA.data_dir_val,
                                        cfg.DATA.data_dir_val_label,
                                        cfg.DATA.data_list_val,
                                        crop_size=(cfg.DATA.input_size_target, cfg.DATA.input_size_target),
                                        stride=cfg.DATA.target_stride)
    valid_provider = torch.utils.data.DataLoader(val_data,
                                        batch_size=1,
                                        shuffle=False)

    target_evaluation = Evaluation(root_label=cfg.DATA.data_dir_val_label,
                                   list_path=cfg.DATA.data_list_val)
    print('Begin inference...')
    f_valid_txt = open(os.path.join(out_path, 'scores.txt'), 'w')
    target_stride = cfg.DATA.target_stride
    preds_int = np.zeros((165, 768, 1024), dtype=np.uint8)
    preds = np.zeros((165, 768, 1024), dtype=np.float32)
    t1 = time.time()
    # for features
    emb_f = []
    emb_b = []
    for i_pic, (cimg, clabel, aimg, alabel, _) in enumerate(valid_provider):
        cimg = cimg.to(device)
        aimg = aimg.to(device)
        img_cat = torch.cat([cimg, aimg], dim=1)
        with torch.no_grad():
            cpred, apred, cfeature, afeature, = model(img_cat, diff=False)
            # mask_f = clabel == 1
            # mask_b = clabel == 0
            # cfeature = cfeature[:, :, 176:-176, 48:-48].permute(0,2,3,1)
            # c_f = cfeature[mask_f, :].cpu().numpy()
            # c_b = cfeature[mask_b, :].cpu().numpy()
            # for i in range(len(c_f)):
            #     emb_f.append(c_f[i])
            # for i in range(len(c_b)):
            #     emb_b.append(c_b[i])
            #
            # if i_pic == 20:
            #
            #     random_f = []
            #     random_b = []
            #     output_f = []
            #     output_b = []
            #     for i in range(250):
            #         random_f.append([np.random.randint(0, len(emb_f))])
            #         random_b.append([np.random.randint(0, len(emb_b))])
            #     for i in range(250):
            #         output_f.append(emb_f[random_f[i][0]])
            #         output_b.append(emb_b[random_b[i][0]])
            #     #
                #
                #
                #
                # mask_f = alabel == 1
                # mask_b = alabel == 0
                # cfeature = cfeature[:, :, 176:-176, 48:-48].permute(0,2,3,1)
                # c_f = afeature[mask_f, :].cpu()
                # c_b = afeature[mask_b, :].cpu()
                # for i in range(len(c_f)):
                #     emb_f.append(c_f[i])
                # for i in range(len(c_b)):
                #     emb_b.append(c_b[i])
        preds_int[i_pic] = inference_results(cpred, preds_int[i_pic])
        preds_int[i_pic+target_stride] = inference_results(apred, preds_int[i_pic+target_stride])
        preds[i_pic] = inference_results2(cpred, preds[i_pic])
        preds[i_pic+target_stride] = inference_results2(apred, preds[i_pic+target_stride])
    t2 = time.time()
    print('Prediction time (s):', (t2 - t1))

    f_out = h5py.File(os.path.join(out_path, 'preds.hdf'), 'w')
    f_out.create_dataset('main', data=preds, dtype=np.float32, compression='gzip')
    f_out.close()

    if args.show:
        print('Show...')
        show_test(preds_int, target_evaluation.get_gt(), cfg.DATA.data_dir_val, seg_img_path)

    if args.mode_map == 'map':
        # mAP, F1, MCC, and IoU
        print('Measure on mAP, F1, MCC, and IoU...')
        t3 = time.time()
        mAP, F1, MCC, IoU = target_evaluation(preds, mode='map')
        t4 = time.time()
        print('mAP=%.4f, F1=%.4f, MCC=%.4f, IoU=%.4f' % (mAP, F1, MCC, IoU))
        print('Measurement time (s):', (t4 - t3))
        f_valid_txt.write('mAP=%.4f, F1=%.4f, MCC=%.4f, IoU=%.4f' % (mAP, F1, MCC, IoU))
        f_valid_txt.write('\n')
    else:
        # dice and jac
        print('Measure on Dice and JAC...')
        mean_dice, mean_jac = target_evaluation(preds_int, mode='dice')
        print('dice=%.6f, jac=%.6f' % (mean_dice, mean_jac))
        f_valid_txt.write('dice=%.6f, jac=%.6f' % (mean_dice, mean_jac))
        f_valid_txt.write('\n')
    f_valid_txt.close()

    print('Done')