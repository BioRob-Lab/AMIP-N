import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self,length_in, feature_in, length_out, ker_size, num_channels, num_hidden, num_heads, dp=0.5):
        super(GRU, self).__init__()

        self.length_out=length_out
        self.num_hidden = num_hidden

        # GRU layers (3 layers)128,128,128
        self.gru1 = nn.GRU(feature_in, num_hidden[0], 1, batch_first=True).double()
        self.gru2 = nn.GRU(num_hidden[0], num_hidden[1], 1, batch_first=True).double()
        self.gru3 = nn.GRU(num_hidden[1], num_hidden[2], 1, batch_first=True).double()

        
        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(7680, 512).double()
        self.fc2 = nn.Linear(512, length_out).double()

    def forward(self, X_cont):
        batch_size = X_cont.size(0)
        seq_len = X_cont.size(1)
        features_len = X_cont.size(2)
        

        # GRU layers
        h_1 = torch.randn(1, batch_size, self.num_hidden[0], dtype=torch.double).to(X_cont.device)
        h_out1, c_out1 = self.gru1(X_cont, h_1)
        # print("GRU 1 Output Size:", h_out1.size())

        h_2 = torch.randn(1, batch_size, self.num_hidden[1], dtype=torch.double).to(X_cont.device)
        h_out2, c_out2 = self.gru2(h_out1, h_2)
        # print("GRU 2 Output Size:", h_out2.size())

        h_3 = torch.randn(1, batch_size, self.num_hidden[2], dtype=torch.double).to(X_cont.device)
        h_out3, c_out3 = self.gru3(h_out2, h_3)
        # print("GRU 3 Output Size:", h_out3.size())

        # Fully connected layers
        fc_out1 = self.fc1(h_out3.reshape(batch_size, -1))
        fc_out2 = self.fc2(fc_out1).view(batch_size, self.length_out, 1)

        return fc_out2