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

class cwner(nn.Module):
    def __init__(self, params):
        super(cwner, self).__init__()
        self.device = params.device
        self.charvocabsize = params.cvocabsize
        self.wordvocabsize = params.wvocabsize
        self.num_layer = params.nl
        self.bidir = params.bidir
        self.emb_size = params.inpsize
        self.bd = 2 if self.bidir else 1
        self.char_encoding = nn.Embedding(self.charvocabsize, self.emb_size) 
        self.word_encoding = nn.Embedding(self.wordvocabsize, self.emb_size)
        #self.char2wordenc = pass

        self.crnn = nn.GRU(self.emb_size, params.hs, \
                          num_layers=self.num_layer, batch_first=True, \
                          bidirectional = self.bidir)
        self.wrnn = nn.GRU(self.emb_size, params.hs, \
                          num_layers=self.num_layer, batch_first=True, \
                          bidirectional = self.bidir)
        self.nernn = nn.GRU(self.emb_size*2, params.hs, \
                          num_layers=self.num_layer, batch_first=True, \
                          bidirectional = self.bidir)

        self.fc = torch.nn.Linear(params.hs,1)


    def forward(self, inp):

        cx, wx, cspans, cmask , wmask= inp
        char_x = self.char_encoding(cx);# print('char-emb:', char_x.shape)
        word_x = self.word_encoding(wx); #print('word-emb:', word_x.shape)

        cemb, _ = self.crnn(char_x); #print('char-from rnn:', cemb.shape)
        wemb, _ = self.wrnn(word_x); #print('word-from rnn:', wemb.shape)

        # c2w should same dim as wemb
        #print(cspans.shape)
        c2w = self.char2wordenc(cemb, cspans, cmask, wmask, wemb)
        # stack word c2w 
        feat_combined = torch.cat( (c2w, wemb),-1)

        rnn_otpt, hn = self.nernn(feat_combined.to(self.device))#, self.h0_new)
        #print('nerrnn output:', rnn_otpt.shape)
        fc_otpt = self.fc(rnn_otpt)
        #print(fc_otpt.shape)
        return fc_otpt

        # input to rnn


    def char2wordenc(self, char_embedding, char_spans, char_mask, wmask, wemb):
        #print('===',char_embedding.shape, char_mask, char_embedding[char_mask].shape)
        first_bpe = char_spans[:,:,0]
        last_bpe = char_spans[:,:,1]
        #import pdb; pdb.set_trace()
        n_bpe = last_bpe - first_bpe
        #print('nbpe shape', n_bpe.shape)
        mask = n_bpe.ne(0)
        #print(mask)
        n_bpe = n_bpe[mask]
        #print(n_bpe)
        indices = torch.arange(n_bpe.size(0), device=self.device).repeat_interleave(n_bpe)
        #print('\n\n',indices)
        #print()
        average_vectors = torch.zeros(n_bpe.size(0), char_embedding.size(-1), device=self.device)
        #print('initial avg vec shape:', average_vectors.shape)
        #print(len(indices), len(char_embedding[char_mask]))
        average_vectors.index_add_(0, indices, char_embedding[char_mask].to(device=self.device))
        average_vectors.div_(n_bpe.view(n_bpe.size(0),1))
        #padding = (average_vectors.shape, output_[wmask].size(0) - average_vectors.size(0))
        #average_vectors = torch.vstack((average_vectors,torch.zeros(padding, char_embedding.size(-1))))
        output_ = torch.zeros_like(wemb)#; print(average_vectors.shape, output_[wmask].shape)
        padding = output_[wmask].size(0) - average_vectors.size(0)
        average_vectors = torch.vstack((average_vectors,torch.zeros(padding, char_embedding.size(-1)).to(self.device))).to(self.device)
        output_[wmask] = average_vectors

        return output_

