import os
import time
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel#, AutoModelForMaskedLM

from encoder import BertEncoder

class charner(torch.nn.Module):
    def __init__(self, params):
        super(charner, self).__init__()
        self.device = params.device
        self.hiddensize = params.hs
        self.num_layer = params.nl
        self.bidir = params.bidir
        self.do_attn = params.do_attn
        self.natth = params.natth
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
        #self.num_atth = params.num_atth
        self.attn_dim = 2*self.hiddensize if self.bidir else 1*self.hiddensize
        self.multihead_attn = nn.MultiheadAttention(self.attn_dim, num_heads=self.natth, batch_first=True)
        
        
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


class bertner(torch.nn.Module):
    def __init__(self, params):
        super(bertner, self).__init__()
        self.device = params.device
        self.hiddensize = params.hs
        self.num_layer = params.nl
        self.bidir = params.bidir
        self.do_attn = params.do_attn
        self.natth = params.natth
        self.bs = params.bs
        self.inplen = params.inplen
        self.inpsize = params.inpsize
        self.vocabsize = params.vocabsize
        self.bd = 2 if self.bidir else 1
        """
        twitter-roberta : 50265 features
        """
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        self.bertmodel = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        self.encoder = BertEncoder(self.bertmodel, self.tokenizer)
        # keep batch first
        self.rnn = nn.GRU(self.inpsize, params.hs, \
                          num_layers=self.num_layer, batch_first=True, \
                          bidirectional = self.bidir)
        #self.num_atth = params.num_atth
        self.attn_dim = 2*self.hiddensize if self.bidir else 1*self.hiddensize
        self.multihead_attn = nn.MultiheadAttention(self.attn_dim, num_heads=self.natth, batch_first=True)
        
        
        self.fc = torch.nn.Linear(self.bd * self.hiddensize, 1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, inp):
        #print('x_i:',x.shape, 'h0:',self.h0.shape)
        #attm, x, span = inp
        self.h0 = torch.randn(self.bd * self.num_layer, 
                              inp[0].shape[0], self.hiddensize, device=self.device)
        bert_x = self.encoder(inp)
        #print(bert_x.shape)
        #bert_x = self.encoder(encoded_x)

        rnn_otpt, hn = self.rnn(bert_x.to(self.device), self.h0)

        if self.do_attn:
            att_otpt, att_wts = self.multihead_attn(rnn_otpt, rnn_otpt, rnn_otpt)
            fc_otpt = self.fc(att_otpt)
        else:
            #print(rnn_otpt.shape)
            fc_otpt = self.fc(rnn_otpt)#self.relu(self.fc(rnn_otpt))

        #print('o shape:',fc_otpt.shape)
        return fc_otpt 
