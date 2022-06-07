#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import pprint as pp
from tqdm import tqdm
import ipdb
import multiprocessing as mp
import os
import h5py
import argparse

import numpy as np
import scipy
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataset import CohenDataSet
from model import CohenConvNet

import tools
import tools_plot


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def F_do_norm_meanstd(data_m):
    """
    description:
        perform z-norm (mean/std)
    inputs:
        - data_m (nb_dim, nb_frame)
    outputs:
        - data_m (nb_dim, nb_frame)
    """
    #m_v = np.mean(data_m, axis=1, keepdims=True)
    #m_m = np.repeat(m_v, data_m.shape[1], axis=1)
    #s_v = np.std(data_m, axis=1, keepdims=True) + 1e-16
    #s_m = np.repeat(s_v, data_m.shape[1], axis=1)
    #out = (data_m-m_m)/s_m
    out = scaler.fit_transform(data_m.T).T
    return out




def F_convert_npz_to_hdf5(pyjama_entry_l, data_dir, dataset_hdf5):
    """
    description:
        convert set of .npz file to a single hdf5 file
    inputs:
        -
    outputs:
        -
    """
    do_norm = True
    
    if os.path.isfile(dataset_hdf5): 
        os.remove(dataset_hdf5)
    hdf5_fid = h5py.File(dataset_hdf5, 'a')
    
    for idx_file, entry in tqdm(enumerate(pyjama_entry_l)):
        audio_file = entry['filepath'][0]['value']
        out_file = tools.F_get_filename(audio_file, data_dir)
        data = np.load(out_file)
        LMS_data_m = data['LMS_data_m']
        SSMmfcc_data_m = data['SSMmfcc_data_m']
        SSMchroma_data_m = data['SSMchroma_data_m']
        # --- z-norm features
        if do_norm:
            LMS_data_m = F_do_norm_meanstd(LMS_data_m)
            SSMmfcc_data_m = F_do_norm_meanstd(SSMmfcc_data_m)
            SSMchroma_data_m = F_do_norm_meanstd(SSMchroma_data_m)
        hdf5_fid[f'/{idx_file}/LMS_data_m'] = LMS_data_m
        hdf5_fid[f'/{idx_file}/SSMmfcc_data_m'] = SSMmfcc_data_m
        hdf5_fid[f'/{idx_file}/SSMchroma_data_m'] = SSMchroma_data_m
        hdf5_fid[f'/{idx_file}/time_sec_v'] = data['LMS_time_sec_v']
    
    hdf5_fid.close()
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='music boundary detection')
    parser.add_argument('--data-dir', default='./data/',
                        help='folder where audio feature files are stored')
    parser.add_argument('--figure-dir', default='./fig/',
                        help='folder where to store the figures')
    args = parser.parse_args()
    
    # --- Set Training and Test set
    pyjama_file = './rwc-pop.pyjama'
    with open(pyjama_file, 'r') as fid: 
        pyjama_data = json.load(fid)
    
    train_entry_l = [entry for idx, entry in enumerate(pyjama_data['collection']['entry']) if (idx%10) != 0]
    test_entry_l = [entry for idx, entry in enumerate(pyjama_data['collection']['entry']) if (idx%10) == 0]
    
    train_set_hdf5 = './rwc-pop_train.hdf5'
    test_set_hdf5 = './rwc-pop_test.hdf5'
    #F_convert_npz_to_hdf5(train_entry_l, args.data_dir, train_set_hdf5)
    #F_convert_npz_to_hdf5(test_entry_l, args.data_dir, test_set_hdf5)

    train_dataset = CohenDataSet(train_entry_l, train_set_hdf5)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    test_dataset = CohenDataSet(test_entry_l, test_set_hdf5)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    nb_epoch = 50
    model = CohenConvNet().cuda()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loss_l = []
    test_loss_l = []
    for num_epoch in range(nb_epoch):    
            # --- Train
            model.train()
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                LMS_data_m = batch[0][:,None,:,:]
                SSM_data_m = torch.cat((batch[1][:,None,:,:], batch[2][:,None,:,:]), 1)
                is_boundary = batch[3]

                model.zero_grad()
                hat_is_boundary = model(LMS_data_m, SSM_data_m)
                loss = criterion(hat_is_boundary, is_boundary)
                loss.backward()
                optimizer.step()
            print(f'epoch: {num_epoch} train_loss: {loss.item()}')
            train_loss_l.append(loss.item())

            # --- Eval
            model.eval()
            for batch_idx, batch in enumerate(test_loader):
                LMS_data_m = batch[0][:,None,:,:]
                SSM_data_m = torch.cat((batch[1][:,None,:,:], batch[2][:,None,:,:]), 1)
                is_boundary = batch[3]

                hat_is_boundary = model(LMS_data_m, SSM_data_m)
                loss = criterion(hat_is_boundary, is_boundary)
            print(f'\tepoch: {num_epoch} test_loss: {loss.item()}')
            test_loss_l.append(loss.item())

            tools_plot.F_test_onefile(test_dataset, num_epoch, model, args.figure_dir)