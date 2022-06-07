import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def F_test_onefile(test_dataset, num_epoch, model, figure_dir):
    """
    description:
        apply model to all temporal patches of a given file
    """

    idx_file_l = [patch_info['idx_file'] for patch_info in test_dataset.patch_info_l]
    
    plt.figure()
    for idx, idx_file in enumerate([0,1,2,3]):
        pos_v = np.where(np.asarray(idx_file_l)==idx_file)[0]
        # ---------------------------------
        is_boundary_l = []
        hat_is_boundary_l = []
        for pos in pos_v:
            batch = test_dataset[pos]
            LMS_data_m = batch[0][None,None,:,:]
            SSM_data_m = torch.cat((batch[1][None,None,:,:], batch[2][None,None,:,:]), 1)
            is_boundary = batch[3]
            hat_is_boundary = model(LMS_data_m, SSM_data_m)
            is_boundary_l.append( is_boundary.cpu().item())
            hat_is_boundary_l.append( hat_is_boundary.cpu().item())
        # ---------------------------------
        plt.subplot(2,2,idx+1)
        plt.plot(is_boundary_l, 'g')
        plt.plot(hat_is_boundary_l, 'r')
    
    # --- create folder if does not exist
    if not os.path.exists(figure_dir): os.makedirs(figure_dir)     
    plt.savefig(f'{figure_dir}/fig_boundary_{idx_file}_{num_epoch}.png', dpi=100)
    plt.close()