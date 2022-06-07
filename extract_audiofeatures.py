#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import multiprocessing as mp
import ipdb
import tqdm

import numpy as np
import scipy
import librosa

import tools


def F_subsample(data_m, time_sec_v, factor):
    """
    description:
        perform max-pooling by a factor 'factor' over the column dimension
    inputs:
        - data_m (nb_dim, nb_frame)
    outputs:
        - data_m (nb_dim, nb_frame/factor)
    """
    nb_dim = data_m.shape[0]
    nb_frame = data_m.shape[1]
    range_v = np.arange(0, nb_frame, factor)
    sub_data_m = np.zeros((nb_dim, len(range_v)))
    sub_time_sec_v = np.zeros(len(range_v))
    for idx, num_frame in enumerate(range_v):
        sub_data_m[:,idx] = np.max(data_m[:,num_frame:num_frame+factor], axis=1)
        sub_time_sec_v[idx] = np.mean(time_sec_v[num_frame:num_frame+factor])
    return sub_data_m, sub_time_sec_v



def F_subsample2D(data_m, time_sec_v, factor):
    """
    description:
        perform max-pooling by a factor 'factor' over both row and column dimensions
    inputs:
        - data_m (nb_frame, nb_frame)
    outputss:
        - data_m (nb_frame/factor, nb_frame/factor)
    """
    nb_frame = data_m.shape[1]
    range_v = np.arange(0, nb_frame, factor)
    sub_data_m = np.zeros((len(range_v), len(range_v)))
    sub_time_sec_v = np.zeros(len(range_v))
    for idx, num_frame in enumerate(range_v):
        for iidx, nnum_frame in enumerate(range_v):
            sub_data_m[idx, iidx] = np.max(data_m[num_frame:num_frame+factor,nnum_frame:nnum_frame+factor])
        sub_time_sec_v[idx]= np.mean(time_sec_v[num_frame:num_frame+factor])
    return sub_data_m, sub_time_sec_v



def F_get_audio_features(inputs):
    """
    description:
        extract audio features (Log-Mel-Spectrogram, MFCC-Self-Similarity-Matrix, Chroma-Self-Similarity-Matrix) for a given file and 
        save results in a .npz file
    inputs:
        - audio_file: full path to an audio file
        - data_dir: folder where to write the .npz file
    outputs:
        -
    """
    audio_file = inputs[0]
    data_dir = inputs[1]

    print(f'computing {audio_file}')
    audio_v, sr_hz = librosa.load(audio_file)
    
    # ----------------
    # --- LMS
    # ----------------
    MS_data_m = librosa.feature.melspectrogram(y=audio_v, sr=sr_hz, window='hanning', win_length=1024, hop_length=512, n_mels=80)
    #LMS_data_m = librosa.power_to_db(MS_data_m,ref=np.max)
    C = 10000
    LMS_data_m = np.log(1 + C*MS_data_m) - np.log(1 + C)
    LMS_time_sec_v = librosa.frames_to_time(frames=np.arange(0, LMS_data_m.shape[1]),  hop_length=512)
    # --- Downsample by 6
    subLMS_data_m, subLMS_time_sec_v = F_subsample(LMS_data_m, LMS_time_sec_v, 6)

    # ----------------
    # --- MFCC
    # ----------------
    # --- Downsample by 2
    LMS_data_m, LMS_time_sec_v = F_subsample(LMS_data_m, LMS_time_sec_v, 2)
    # --- DCT
    MFCC_data_m = scipy.fftpack.dct(LMS_data_m, axis=0, type=2, norm='ortho')
    # --- omit 0th coefficient
    MFCC_data_m = MFCC_data_m[1:,:]
    # --- stacked by 2
    MFCC_data_m = np.concatenate((MFCC_data_m[:,0:-1], MFCC_data_m[:,1:]), axis=0)
    MFCC_time_sec_v = LMS_time_sec_v[0:-1]
    # --- Compute SSM
    MFCC_data_m /= (np.sqrt(np.sum(MFCC_data_m**2, axis=0)) + 1e-16)
    SSMmfcc_data_m = 1 - (MFCC_data_m.T @ MFCC_data_m)
    # --- Downsmaple by 3
    subSSMmfcc_data_m, subSSMmfcc_time_sec_v = F_subsample2D(SSMmfcc_data_m, MFCC_time_sec_v, 3)

    # ----------------
    # --- Chroma
    # ----------------
    SPEC_data_m = np.abs(librosa.stft(y=audio_v, window='hanning', win_length=1024, hop_length=512))**2
    SPEC_time_sec_v = librosa.frames_to_time(frames=np.arange(0, SPEC_data_m.shape[1]),  hop_length=512)
    # --- Downsample by 2
    SPEC_data_m, SPEC_time_sec_v = F_subsample(SPEC_data_m, SPEC_time_sec_v, 2)
    # --- Chroma
    CHROMA_data_m = librosa.feature.chroma_stft(S=SPEC_data_m, sr=sr_hz)
    CHROMA_time_sec_v = SPEC_time_sec_v
    # --- stacked by 2
    CHROMA_data_m = np.concatenate((CHROMA_data_m[:,0:-1], CHROMA_data_m[:,1:]), axis=0)
    CHROMA_time_sec_v = CHROMA_time_sec_v[0:-1]
    # --- Compute SSM
    CHROMA_data_m /= (np.sqrt(np.sum(CHROMA_data_m**2, axis=0)) + 1e-16)
    SSMchroma_data_m = 1 - (CHROMA_data_m.T @ CHROMA_data_m)
    # --- Downsmaple by 3
    subSSMchroma_data_m, subSSMchroma_time_sec_v = F_subsample2D(SSMchroma_data_m, CHROMA_time_sec_v, 3)

    out_file = tools.F_get_filename(audio_file, data_dir)
    np.savez(out_file, 
            LMS_data_m=subLMS_data_m, LMS_time_sec_v=subLMS_time_sec_v, 
            SSMmfcc_data_m=subSSMmfcc_data_m, SSMmfcc_time_sec_v=subSSMmfcc_time_sec_v,
            SSMchroma_data_m=subSSMchroma_data_m, SSMchroma_time_sec_v=subSSMchroma_time_sec_v,
            )
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='music boundary detection')
    parser.add_argument('--pyjama-file', default='./rwc-pop.pyjama',
                        help='pyjama file containing dataset description')
    parser.add_argument('--data-dir', default='./data/',
                        help='folder where to store the audio feature files')
    args = parser.parse_args()
    
    # ----------------------------
    # --- Compute audio features
    # ----------------------------
    with open(args.pyjama_file, 'r') as fid: 
        pyjama_data = json.load(fid)

    audio_file_l = [entry['filepath'][0]['value'] for entry in pyjama_data['collection']['entry']]
    to_compute_l = []
    for audio_file in audio_file_l:
        out_file = tools.F_get_filename(audio_file, args.data_dir)
        if not os.path.isfile(out_file): to_compute_l.append(audio_file)
    print(f'number of remaining files to process: {len(to_compute_l)}/{len(audio_file_l)}')
    to_compute_l = [(to_compute, args.data_dir) for to_compute in to_compute_l]
    # --- Compute
    a_pool = mp.Pool()
    a_pool.map(F_get_audio_features, to_compute_l)
    #for to_compute in to_compute_l:
    #    F_get_audio_features(to_compute)