import torch
import torch.nn as nn
import models.TCN.tcn_my as tcn
from models.TCN.tcn_my import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, length_in, features_in, length_out,num_channels, ker_size, dp=0.3):
        super(TCN, self).__init__()

        self.length_out=length_out
        
        # Original TemporalConvNet with 1 layer
        self.tcn1 = TemporalConvNet(length_in, [num_channels], kernel_size=ker_size, dropout=dp).double()

        # Additional TemporalBlocks (2 layers)
        self.tcn2 = TemporalConvNet(num_channels, [num_channels], kernel_size=ker_size, dropout=dp).double()
        self.tcn3 = TemporalConvNet(num_channels, [num_channels], kernel_size=ker_size, dropout=dp).double()
        self.tcn4 = TemporalConvNet(num_channels, [num_channels], kernel_size=ker_size, dropout=dp).double()
        self.tcn5 = TemporalConvNet(num_channels, [num_channels], kernel_size=ker_size, dropout=dp).double()

        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(10240, 512).double()
        self.fc2 = nn.Linear(512, length_out).double()

    def forward(self, X_cont):
        batch_size = X_cont.size(0)
        q_len = X_cont.size(1)
        features_len = X_cont.size(2)

        # TCN layers
        out_tcn1 = self.tcn1(X_cont)
        # print("TCN 1 Output Size:", out_tcn1.size())
        out_tcn2 = self.tcn2(out_tcn1)
        # print("TCN 2 Output Size:", out_tcn2.size())
        out_tcn3 = self.tcn3(out_tcn2)
        # print("TCN 3 Output Size:", out_tcn3.size())

    
        # Fully connected layers
        fc_out1 = self.fc1(out_tcn3.reshape(batch_size, -1))
        fc_out2 = self.fc2(fc_out1).view(batch_size,self.length_out,1)


        return fc_out2