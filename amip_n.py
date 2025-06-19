import torch
import torch.nn as nn
import models.TCN.tcn_my as tcn
from models.TCN.tcn_my import TemporalConvNet
from models.attention.Wavelet_Multi_Head_Self_Attention import WaveletMultiHeadAttention
from models.attention.attn import ProbAttention,LogSparceAttention,FourierProbAttention, MultiWaveletProbAttention,AttentionLayer
from models.attention.DCTMutiHeadSelfAttention import DiscreteCosineMultiHeadSelfAttention
from models.attention.FourierMutiHeadSelfAttention import FourierMultiHeadSelfAttention
from models.attention.WaveletMutiHeadSelfAttention import MultiWaveletMutiHeadSelfAttention
from models.attention.FourierCorrelation import FourierCrossAttention,AutoCorrelationLayer

class AMIP_N(nn.Module):
    def __init__(self, length_in, feature_in, length_out, ker_size, num_channels, num_hidden, num_heads, dp=0.3):
        super(AMIP_N, self).__init__()

        self.num_hidden = num_hidden
        self.num_channels = num_channels
        self.length_out=length_out

        # Original TemporalConvNet with 1 layer
        self.tcn1 = TemporalConvNet(length_in, [num_channels[0]], kernel_size=ker_size, dropout=dp).double()

        # Additional TemporalBlocks (2 layers)
        self.tcn2 = TemporalConvNet(num_channels[0], [num_channels[1]], kernel_size=ker_size, dropout=dp).double()
        self.tcn3 = TemporalConvNet(num_channels[1], [num_channels[2]], kernel_size=ker_size, dropout=dp).double()
        self.dropout_tcn3 = nn.Dropout(dp)
        self.attention = WaveletMultiHeadAttention(feature_in, num_heads)

        # self.attention = DiscreteCosineMultiHeadSelfAttention(in_features=feature_in, kernel_size=40, d_model=40, num_heads=num_heads, dct_n=40, dropout=dp)

        # self_att = FourierCrossAttention(in_channels=feature_in,out_channels=feature_in,seq_len_q=length_in,seq_len_kv=length_in,modes=64,mode_select_method='random',activation='tanh',policy=0)

        # self.attention = AutoCorrelationLayer(self_att,d_model=feature_in,n_heads=num_heads)        
        
        # LSTM layers (3 layers)128,128,128
        self.lstm1 = nn.LSTM(feature_in, num_hidden[0], 1, batch_first=True, dropout=dp).double()
        self.lstm2 = nn.LSTM(num_hidden[0], num_hidden[1], 1, batch_first=True, dropout=dp).double()
        self.lstm3 = nn.LSTM(num_hidden[1], num_hidden[2], 1, batch_first=True, dropout=dp).double()
        self.dropout_lstm1 = nn.Dropout(dp)
                     
        # Fully connected layers (3 layers)
        self.fc1 = nn.Linear(num_hidden[1] * num_hidden[2], 512).double()
        self.fc2 = nn.Linear(512, length_out).double()
        
        

    def forward(self, X_cont):
        batch_size = X_cont.size(0)
        seq_len = X_cont.size(1)
        features_len = X_cont.size(2)
       
        
        
        
        # TCN layers
        out_tcn1 = self.tcn1(X_cont)
        # print("TCN 1 Output Size:", out_tcn1.size())
        out_tcn2 = self.tcn2(out_tcn1)
        # print("TCN 2 Output Size:", out_tcn2.size())
        
        out_tcn3 = self.tcn3(out_tcn2)
        # print("TCN 3 Output Size:", out_tcn3.size())
        out_tcn3 = self.dropout_tcn3(out_tcn3)
        attention_output = self.attention(out_tcn3)  # Adjust input permutation
        
    

        # LSTM layers
        h_1 = torch.randn(1, batch_size, self.num_hidden[0], dtype=torch.double).to(X_cont.device)
        c_1 = torch.randn(1, batch_size, self.num_hidden[0], dtype=torch.double).to(X_cont.device)
        h_out1, c_out1 = self.lstm1(attention_output, (h_1, c_1))
        # print("LSTM 1 Output Size:", h_out1.size())
        
        h_2 = torch.randn(1, batch_size, self.num_hidden[1], dtype=torch.double).to(X_cont.device)
        c_2 = torch.randn(1, batch_size, self.num_hidden[1], dtype=torch.double).to(X_cont.device)
        h_out2, c_out2 = self.lstm2(h_out1, (h_2, c_2))
        # print("LSTM 2 Output Size:", h_out2.size())
        
        h_3 = torch.randn(1, batch_size, self.num_hidden[2], dtype=torch.double).to(X_cont.device)
        c_3 = torch.randn(1, batch_size, self.num_hidden[2], dtype=torch.double).to(X_cont.device)
        h_out3, c_out3 = self.lstm3(h_out2, (h_3, c_3))
        # print("LSTM 3 Output Size:", h_out3.size())
        h_out3 = self.dropout_lstm1(h_out3)
        
        # Fully connected layers
        fc_out1 = self.fc1(h_out3.reshape(batch_size, -1))
        fc_out2 = self.fc2(fc_out1).view(batch_size,self.length_out,1)

        return fc_out2