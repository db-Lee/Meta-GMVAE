import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OmniglotEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(OmniglotEncoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size

        self.encoder = nn.Sequential(
            # -1, hidden_size, 14, 14
            nn.Conv2d(1, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, hidden_size, 7, 7
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, 2*hidden_size, 4, 4
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            # -1, 2*hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True)
        )

    def forward(self, x):
        # -1, hidden_size, 2, 2
        h = self.encoder(x)
        # -1, hidden_size*2*2
        h = h.view(-1, self.last_hidden_size)
        return h

class OmniglotDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(OmniglotDecoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.decoder = nn.Sequential(
            # -1, hidden_size, 4, 4
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.ConvTranspose2d(hidden_size, hidden_size, 3, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 14, 14
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, 1, 28, 28
            nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1),
            nn.Sigmoid()
        )        

    def forward(self, h):
        # -1, hidden_size, 2, 2
        h = h.view(-1, self.hidden_size, 2, 2)
        # -1, 1, 28, 28
        x = self.decoder(h)
        return x

# The below code is adapted from github.com/juho-lee/set_transformer/blob/master/modules.py 
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        #O = O + F.elu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)
