import torch
import torch.nn as nn
import torch.nn.functional as F


class CohenConvNet(nn.Module):
    def __init__(self):
        super(CohenConvNet, self).__init__()
        """
        """

        # --- L (Left) R (Right) M (middle) branch number of channels
        L_C = 16
        R_C = 16
        M_C = 32
        self.L = nn.Sequential(
            nn.Conv2d(1, L_C, kernel_size = (17, 9), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (4, 6))
        )
        self.R = nn.Sequential(
            nn.Conv2d(2, R_C, kernel_size = (9, 9), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (6, 6))
        )
        
        self.M_CNN = nn.Conv2d(L_C+R_C, M_C, kernel_size = (7, 7), stride=(1, 1))
        
        self.M_FC = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=M_C*10*10, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, LMS_data_m, SSM_data_m):
        """
        inputs:
        - LMS_data_m (m_batch, 1, 80, 104)
        - SSM_data_m (m_batch, 2, 104, 104)
        """
        m_batch =  LMS_data_m.size()[0]

        # --- lll (m, L_C, 16, 16)
        lll = self.L( LMS_data_m )
        # --- rrr (m, R_C, 16, 16)
        rrr = self.R( SSM_data_m )
        
        # --- mmm1 (m_batch, L_C+R_C, 16, 16)
        mmm1 = torch.cat((lll, rrr), 1)
        # --- mmm2 (m_batch, M_c, 10, 10)
        mmm2 = nn.ReLU()(self.M_CNN(mmm1))

        # --- mmm3 (m_batch, 128)
        out = self.M_FC( mmm2.view(m_batch, -1) )
        return out.squeeze()
