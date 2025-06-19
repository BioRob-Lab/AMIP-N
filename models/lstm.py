import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, length_in, feature_in, length_out, ker_size, num_channels, num_hidden, num_heads, dp=0.5):
        super(LSTM, self).__init__()

        self.num_hidden = num_hidden
        self.length_out = length_out
 
        # LSTM layers (3 layers) 100,100
        self.lstm1 = nn.LSTM(feature_in, num_hidden[0], 1, batch_first=True).double()
        self.lstm2 = nn.LSTM(num_hidden[0], num_hidden[0], 1, batch_first=True).double()

        
        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(7680, 512).double()
        self.fc2 = nn.Linear(512, length_out).double()

    def forward(self, X_cont):
        batch_size = X_cont.size(0)
        seq_len = X_cont.size(1)
        features_len = X_cont.size(2)
        n_hidden = [100, 256]

        # LSTM layers
        h_1 = torch.zeros(1, batch_size, self.num_hidden[0], dtype=torch.double).to(X_cont.device)
        c_1 = torch.zeros(1, batch_size, self.num_hidden[0], dtype=torch.double).to(X_cont.device)
        h_out1, c_out1 = self.lstm1(X_cont, (h_1, c_1))
        # print("LSTM 1 Output Size:", h_out1.size())
        
        h_2 = torch.zeros(1, batch_size, self.num_hidden[1], dtype=torch.double).to(X_cont.device)
        c_2 = torch.zeros(1, batch_size, self.num_hidden[1], dtype=torch.double).to(X_cont.device)
        h_out2, c_out2 = self.lstm2(h_out1, (h_2, c_2))
        # print("LSTM 2 Output Size:", h_out2.size())
            

        # Fully connected layers
        fc_out1 = self.fc1(h_out2.reshape(batch_size, -1))
        fc_out2 = self.fc2(fc_out1).view(batch_size, self.length_out, 1)

        return fc_out2