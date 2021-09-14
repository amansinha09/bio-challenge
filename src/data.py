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


###################################################


class biodata2(Dataset):
    def __init__(self, df, vocab=None, max_len=None, name='train', level='char'):
        self.len = len(df)
        self.data = df
        self.level = level
        self.setname = name
        if max_len is None:
            if self.level == 'char':
                self.max_len = max(df.text.apply(lambda x: len(x)).to_numpy())
            else:
                self.max_len = max(df.text.apply(lambda x: len(x.split(' '))).to_numpy())
        else:
            self.max_len = max_len
        #print(f'name: {name}, max_len : {self.max_len}')
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
               'targets' : torch.tensor(self.make_label(sen, start, end), dtype=torch.float64),
               'spans': torch.tensor(self.make_span(sen,pad=True), dtype=torch.long)}
    
    def transform_input(self, sentence, pad=False):
        es = []
        inp = sentence.lower() if self.level == 'char' else sentence.split(' ')
        for e in inp:
            if e in self.vdict:
                es.append(self.vdict[e])
            else:
                es.append(self.vdict['<oov>'])
        diff = 0 if self.max_len<len(es) else self.max_len-len(es)
        diff = 0 if not pad else diff
        es =  es + [1]*diff
        return es[:self.max_len]
    
    def make_label(self,sample, s, e):
        l = len(sample)
        label = np.zeros(self.max_len)
        if self.level == 'char':
            try:
                if s <= l:
                    label[s:e] = 1
            except:
                print('======>',s, l)
                import sys;sys.exit()
        else:
            start = 0
            #print(sample)
            fi, bi = -1, -1
            if s != e:
                for i,w in enumerate(sample.split(' ')):
                    start += len(w) + 1
                    if start >= s:
                        fi = i +1
                        label[fi] = 1
                        if start + len(sample.split(' ')) < e:
                            continue
                        else:
                            break
                '''
                tl = len(sample)
                end = tl
                for i,w in reversed(list(enumerate(sample.split(' ')))):
                    tl -= len(w)
                    if tl <= e:
                        bi = i
                        break
                '''
                #try:
                    #assert(fi == bi)
                #    label[fi] = 1
                #except:
                #    print('======>',sample.split(), fi,bi, '#', s,e)
                #    import sys;sys.exit()
        return label[:self.max_len]
    
    def create_vocab(self, vocab):
        if not vocab:
            iv = {'<oov>', '<pad>'}
            for line in self.data['text'].to_numpy():
                if self.level == 'char':
                    iv |= set(line)
                else:
                    iv |= set(line.split(' '))

            self.vocab = iv
        else:
            iv = vocab
        ivdict = {'<oov>':0, '<pad>':1}
        for e in iv:
            if e not in ivdict:
                ivdict[e] = len(ivdict)
        self.vdict = ivdict
        
        print(f'{self.setname} vocab created!')
    
    def make_span(self, sample, pad=False):
        c = 0
        spans = []
        for i,w in enumerate(sample.split()):
            spans.append([c,c+len(w)])
            c += len(w) +1
        if len(spans) < self.max_len:
            diff = self.max_len - len(spans)
            spans += [[-1,-1]]*diff
        return spans[:self.max_len]

    def __len__(self):
        return self.len
    
    def __name__(self):
        return self.setname
