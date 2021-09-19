import os
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from encoder import BertEncoder


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
    def __init__(self, df, vocab=None, max_len=None, name='train', level='char', use_bert=False):
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
        #self.max_len = 100
        self.use_bert = use_bert
        self.bertmodel = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")#; self.bertmodel.save_pretrained("cardiffnlp/twitter-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")#; self.tokenizer.save_pretrained("cardiffnlp/twitter-roberta-base")
        self.encoder = BertEncoder(self.bertmodel, self.tokenizer)
        
        self.vocab = vocab
        self.vdict = None
        self.create_vocab(vocab)
        #self.data['text'] = self.data['text'].apply(lambda x: x.lower())
        
    def __getitem__(self, index):
        sen = self.data['text'].iloc[index]
        start, end = self.data['start'].iloc[index], self.data['end'].iloc[index]
        start = 0 if start == '-' else int(start)
        end = 0 if end == '-' else int(end)
        if not self.use_bert:
            return {'ids' : torch.tensor(self.transform_input(sen, pad=True), dtype=torch.long),
                   'targets' : torch.tensor(self.make_label(sen, start, end), dtype=torch.float64),
                   'spans': torch.tensor(self.make_span(sen,pad=True), dtype=torch.long)}
                   #'attn_mask': torch.ones(self.max_len)
        else:

            tok_ids, attn_mask, spans = self.encoder.encode(seq=sen.split(' '), pad=self.max_len)
            return {'ids': tok_ids,
                    'bspans': spans,
		    'cspans': torch.tensor(self.make_span(sen,pad=True), dtype=torch.long),
                    'attn_mask': attn_mask,
                    'targets': torch.tensor(self.make_label(sen, start, end), dtype=torch.float64)}

    
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


###################################################

class biodata3(Dataset):
    def __init__(self, df, cvocab=None, wvocab=None, cmax_len=None, wmax_len=None, name='train'):
        self.len = len(df)
        self.data = df
        self.setname = name
        
        self.cmax_len = max(df.text.apply(lambda x: len(x)).to_numpy())
        self.wmax_len = max(df.text.apply(lambda x: len(x.split(' '))).to_numpy())
        
        #print(f'name: {name}, max_len : {self.max_len}')
        #self.max_len = 100
        #self.use_bert = use_bert
        #self.bertmodel = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")#; self.bertmodel.save_pretrained("cardiffnlp/twitter-roberta-base")
        #self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")#; self.tokenizer.save_pretrained("cardiffnlp/twitter-roberta-base")
        #self.encoder = BertEncoder(self.bertmodel, self.tokenizer)
        
        self.cvocab = cvocab
        self.cvdict = None
        self.wvocab = wvocab
        self.wvdict = None
        self.create_vocab(cvocab, wvocab)
        #self.data['text'] = self.data['text'].apply(lambda x: x.lower())
        
    def __getitem__(self, index):
        sen = self.data['text'].iloc[index]
        start, end = self.data['start'].iloc[index], self.data['end'].iloc[index]
        start = 0 if start == '-' else int(start)
        end = 0 if end == '-' else int(end)

        cids, cmask, wids, wmask = self.transform_input(sen, pad=True)

        return {'cids' : torch.tensor(cids, dtype=torch.long),
                'cmask': torch.tensor(cmask, dtype= torch.bool),
                'wids': torch.tensor(wids, dtype=torch.long),
                'wmask': torch.tensor(wmask, dtype=torch.bool),
                   'targets' : torch.tensor(self.make_label(sen, start, end), dtype=torch.float64),
                   'spans': torch.tensor(self.make_span(sen,pad=True), dtype=torch.long)}
                   #'attn_mask': torch.ones(self.max_len)

    
    def transform_input(self, sentence, pad=False):
        ces, wes = [], []
        cmask, wmask = [], []
        inp = sentence 
        for e in inp:
            if e in self.cvdict:
                ces.append(self.cvdict[e])
            else:
                ces.append(self.cvdict['<oov>'])
        cmask = [True]*len(ces)
        diff = 0 if self.cmax_len<len(ces) else self.cmax_len-len(ces)
        diff = 0 if not pad else diff
        ces =  ces + [1]*diff; cmask += [False]*diff

        for e in inp.split(' '):
            if e in self.wvdict:
                wes.append(self.wvdict[e])
            else:
                wes.append(self.wvdict['<oov>'])

        wmask = [True]*len(wes)
        diff = 0 if self.wmax_len<len(wes) else self.wmax_len-len(wes)
        diff = 0 if not pad else diff
        wes =  wes + [1]*diff; wmask += [False]*diff

        
        return ces[:self.cmax_len], cmask[:self.cmax_len], wes[:self.wmax_len], wmask[:self.wmax_len]
    
    def make_label(self,sample, s, e):
        l = len(sample)
        label = np.zeros(self.wmax_len)
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
        return label[:self.wmax_len]
    
    def create_vocab(self, cvocab, wvocab):
        if not cvocab:
            civ = {'<oov>', '<pad>'}
            for line in self.data['text'].to_numpy():
                civ |= set(line)    

            self.cvocab = civ
        else:
            civ = cvocab
        civdict = {'<oov>':0, '<pad>':1}
        for e in civ:
            if e not in civdict:
                civdict[e] = len(civdict)
        self.cvdict = civdict

        if not wvocab:
            wiv = {'<oov>', '<pad>'}
            for line in self.data['text'].to_numpy():
                wiv |= set(line.split(' '))

            self.wvocab = wiv
        else:
            wiv = wvocab
        wivdict = {'<oov>':0, '<pad>':1}
        for e in wiv:
            if e not in wivdict:
                wivdict[e] = len(wivdict)
        self.wvdict = wivdict
        
        print(f'{self.setname} vocab created!')
    
    def make_span(self, sample, pad=False):
        '''c = 0
        spans = []
        for i,w in enumerate(sample.split()):
            spans.append([c,c+len(w)])
            c += len(w) +1
        if len(spans) < self.cmax_len:
            diff = self.cmax_len - len(spans)
            spans += [[-1,-1]]*diff
        '''

        spans = []
        start,end = 0,0
        for i,c in enumerate(sample):
            end += 1
            if c == ' ':
                spans.append([start, end])
                start = end
        spans.append([start, end])
        if len(spans) < self.wmax_len:
            diff = self.wmax_len- len(spans)
            for i in range(diff):
                spans.append([end,end])
        return spans[:self.wmax_len]

    def __len__(self):
        return self.len
    
    def __name__(self):
        return self.setname
