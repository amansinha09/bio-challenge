import os
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn



class charner(torch.nn.Module):
    def __init__(self, params):
        super(charner, self).__init__()
        self.device = params.device
        self.hiddensize = params.hs
        self.num_layer = params.nl
        self.bidir = params.bidir
        self.do_attn = params.do_attn
        self.bs = params.bs
        self.inplen = params.inplen
        self.inpsize = params.inpsize
        self.vocabsize = params.vocabsize
        self.bd = 2 if self.bidir else 1
        self.char_embedding = nn.Embedding(self.vocabsize, self.inpsize)
        # keep batch first
        self.rnn = nn.GRU(params.inpsize, params.hs, \
                          num_layers=self.num_layer, batch_first=True, \
                          bidirectional = self.bidir)
        self.num_atth = params.num_atth
        self.attn_dim = 2*self.hiddensize if self.bidir else 1*self.hiddensize
        self.multihead_attn = nn.MultiheadAttention(self.attn_dim, num_heads=1,dropout=0.3, batch_first=True)
        
        
        self.fc = torch.nn.Linear(self.bd * self.hiddensize, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        #print('x_i:',x.shape, 'h0:',self.h0.shape)
        self.h0 = torch.randn(self.bd * self.num_layer, 
                              x.shape[0], self.hiddensize, device=self.device)
        x = self.char_embedding(x.to(self.device))
        rnn_otpt, hn = self.rnn(x.to(self.device), self.h0)

        if self.do_attn:
            att_otpt, att_wts = self.multihead_attn(rnn_otpt, rnn_otpt, rnn_otpt)
            fc_otpt = self.fc(att_otpt)
        else:
            #print(rnn_otpt.shape)
            fc_otpt = self.fc(rnn_otpt)#self.relu(self.fc(rnn_otpt))

        #print('o shape:',fc_otpt.shape)
        return fc_otpt  
