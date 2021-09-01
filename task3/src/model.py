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
        self.bs = params.bs
        self.inplen = params.inplen
        self.inpsize = params.inpsize
        self.vocabsize = params.vocabsize
        
        self.char_embedding = nn.Embedding(self.vocabsize, self.inpsize)
        # keep batch first
        self.rnn = nn.GRU(params.inpsize, params.hs, \
                          num_layers=self.num_layer, batch_first=True)
        
        self.fc = torch.nn.Linear(self.bidir * self.hiddensize, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        #print('x_i:',x.shape, 'h0:',self.h0.shape)
        self.h0 = torch.randn(self.bidir * self.num_layer, 
                              x.shape[0], self.hiddensize, device=self.device)
        x = self.char_embedding(x.to(self.device))
        rnn_otpt, hn = self.rnn(x.to(self.device), self.h0)
        #print(rnn_otpt.shape)
        fc_otpt = self.relu(self.fc(rnn_otpt))
        #print('o shape:',fc_otpt.shape)
        return fc_otpt  
