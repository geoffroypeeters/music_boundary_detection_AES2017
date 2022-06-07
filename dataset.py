import os
import h5py
import ipdb

import numpy as np

import torch
from torch.utils.data import Dataset


import tools

def F_slice_into_patches(patch_hop_frame, patch_halfduration_frame, nb_frame, idx_file):
    """
    description:
        create structure for storing patch-based slides of the spectrogram
    inputs:
        - 
    outputs:
        -
    """
    patch_info_l = []
    middle_frame = patch_halfduration_frame
    while middle_frame + patch_halfduration_frame < nb_frame-1:
        start_frame = middle_frame - patch_halfduration_frame
        stop_frame = middle_frame + patch_halfduration_frame

        patch_info_d = {'idx_file': idx_file, 
                        'start_frame': start_frame, 
                        'middle_frame': middle_frame, 
                        'end_frame': stop_frame}
        patch_info_l.append(patch_info_d)

        middle_frame += patch_hop_frame
    return patch_info_l







class CohenDataSet(Dataset):
    def __init__(self, pyjama_entry_l, dataset_hdf5):
        """
        """
        self.patch_info_l = []
        self.is_boundary_l = []
        self.do_data_in_gpu = True
        self.data_d = {}
        
        patch_halfduration_frame = 52
        patch_hop_frame = int(patch_halfduration_frame/32)
        param_sigma = 1
        
        self.hdf5_fid = h5py.File(dataset_hdf5, 'r')
                
        for idx_file, entry in enumerate(pyjama_entry_l):
            audio_file = entry['filepath'][0]['value']
            nb_frame = self.hdf5_fid[f'/{idx_file}/LMS_data_m'].shape[1]
            # --- display info
            if idx_file==0:
                print(f"patch_duration_sec: {np.mean(np.diff( self.hdf5_fid[f'/{idx_file}/time_sec_v'][:] ))*patch_halfduration_frame*2}")
                print(f"path_hop_sec: {np.mean(np.diff(  self.hdf5_fid[f'/{idx_file}/time_sec_v'][:] ))*patch_hop_frame}")
            print(audio_file)
            
            # --- get patches
            patch_info_l = F_slice_into_patches(patch_hop_frame, patch_halfduration_frame, nb_frame, idx_file)
            self.patch_info_l += patch_info_l
            
            # --- create ground-truth for each patch
            boundary_sec_v = np.asarray([seg['time'] for seg in entry['structtype']])
            for patch in patch_info_l:
                center_time_sec = self.hdf5_fid[f'/{idx_file}/time_sec_v'][patch['middle_frame']]
                # --- get the distance to closest boundary
                delta = np.min(np.abs(center_time_sec-boundary_sec_v))
                # --- assign the value of a Gaussian centered on the closest annotation and evaluated at the center frame
                value = np.exp(-np.power(delta, 2.) / (2 * np.power(param_sigma, 2.)))
                self.is_boundary_l.append(value)
            
            if self.do_data_in_gpu:
                self.data_d[idx_file] = {
                    'LMS_data_m': torch.from_numpy( self.hdf5_fid[f'/{idx_file}/LMS_data_m'][:] ).float().cuda(),
                    'SSMmfcc_data_m': torch.from_numpy( self.hdf5_fid[f'/{idx_file}/SSMmfcc_data_m'][:] ).float().cuda(),
                    'SSMchroma_data_m': torch.from_numpy( self.hdf5_fid[f'/{idx_file}/SSMchroma_data_m'][:] ).float().cuda(),
                    'time_sec_v': torch.from_numpy( self.hdf5_fid[f'/{idx_file}/time_sec_v'][:] ).float().cuda()
                    }


    def __len__(self):
        return len(self.patch_info_l)


    def __getitem__(self, idx_patch):
        idx_file = self.patch_info_l[idx_patch]['idx_file']
        s = self.patch_info_l[idx_patch]['start_frame']
        e = self.patch_info_l[idx_patch]['end_frame']
        if self.do_data_in_gpu:
            LMS_data_m = self.data_d[idx_file]['LMS_data_m'][:,s:e]
            SSMmfcc_data_m = self.data_d[idx_file]['SSMmfcc_data_m'][s:e,s:e]
            SSMchroma_data_m = self.data_d[idx_file]['SSMchroma_data_m'][s:e,s:e]
        else:
            LMS_data_m = torch.from_numpy( self.hdf5_fid[f'/{idx_file}/LMS_data_m'][:,s:e] ).float()
            SSMmfcc_data_m = torch.from_numpy( self.hdf5_fid[f'/{idx_file}/SSMmfcc_data_m'][s:e,s:e] ).float()
            SSMchroma_data_m = torch.from_numpy( self.hdf5_fid[f'/{idx_file}/SSMchroma_data_m'][s:e,s:e] ).float()

        is_boundary = torch.tensor(self.is_boundary_l[idx_patch]).float().cuda()
        return LMS_data_m, SSMmfcc_data_m, SSMchroma_data_m, is_boundary
