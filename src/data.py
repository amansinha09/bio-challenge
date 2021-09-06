import os
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader



class biodata(Dataset):
    def __init__(self, df, vocab=None, name='train'):
        self.len = len(df)
        self.data = df
        self.max_len = max(df.text.apply(lambda x: len(x)).to_numpy())
        self.setname = name
        self.vocab = vocab
        self.vdict = None
        self.create_vocab(vocab)
        #self.data['text'] = self.data['text'].apply(lambda x: x.lower())
        
    def __getitem__(self, index):
        sen = self.data['text'].iloc[index]
        start, end = self.data['start'].iloc[index], self.data['end'].iloc[index]
        start = 0 if start == '-' else int(start)
        end = 0 if end == '-' else int(end)
        return {'ids' : torch.tensor(self.transform_input(sen, pad=True), dtype=torch.long),
               'targets' : torch.tensor(self.make_label(self.max_len, start, end), dtype=torch.float64)}
    
    def transform_input(self, sentence, pad=False):
        es = []
        for e in sentence.lower():
            if e in self.vdict:
                es.append(self.vdict[e])
            else:
                es.append(self.vdict['<oov>'])
        diff = 0 if self.max_len<len(es) else self.max_len-len(es)
        diff = 0 if not pad else diff
        return es + [1]*diff
    
    def make_label(self,l, start, end):
        label = np.zeros(l)
        try:
            if start <= l:
                label[start:end] = 1
        except:
            print('======>',start, l)
            import sys;sys.exit()
        return label
    
    def create_vocab(self, vocab):
        if not vocab:
            iv = {'<oov>', '<pad>'}
            for line in self.data['text'].to_numpy():
                iv |= set(line)
            self.vocab = iv
        else:
            iv = vocab
        ivdict = {'<oov>':0, '<pad>':1}
        for e in iv:
            if e not in ivdict:
                ivdict[e] = len(ivdict)
        self.vdict = ivdict
        
        print(f'{self.setname} vocab created!')
    
    def __len__(self):
        return self.len
    
    def __name__(self):
        return self.setname
