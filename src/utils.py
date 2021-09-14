import os
import sys
import pathlib
import logging as log
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
#from torch.utils.data import Dataset, DataLoader

from data import *
from model import *



def create_pred_file(ddf, output, sps, name, save_dir, level = 'char'):
	# added spans argument : sps
	predques = ddf.iloc[:, 0:4].copy()

	spans = []
	if level == 'char':
		for t in output:
			span = []
			start,end = -1,-1
			for i,tt in enumerate(t):
				if start == -1 and tt == 1:
					start,end = i,i
				if start != -1:
					if tt == 1:
						end +=1
					else:
						span.append((start,end))
						start, end = -1,-1
			spans.append(span)
	else:
		# level ==  'word'
		# reformat spans
		sps = [[interval(s) for s in sl] for sl in list_of_spans]
		for s,l in zip(sps, output):
			assert(len(s) == len(l))
			a,b = recreate(s[l.astype(bool)])
			spans.append((a,b))

	rest_columens = []
	for i,sp in enumerate(spans):
		#print(i,sp)
		if len(sp) == 0:
			k =('-','-','-','-')    
		else:
			# one span detected else first*
			if len(sp) >= 1:
				s = ddf.iloc[i]['text']
				wrd = s[sp[0][0]:sp[0][1]]
				k = (*(sp[0]),wrd, wrd.lower())
		rest_columens.append(k)
		
	predans = pd.DataFrame(rest_columens, columns=['start', 'end', 'span', 'drug'])      
	pred = pd.concat([predques.reset_index(drop=True), predans.reset_index(drop=True)], axis = 1, )
	pred.to_csv(f'{save_dir}/pred_{name}.csv', sep='\t')
	print('predictions saved !!')


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.inf
		self.delta = delta
		self.save_path = save_path
		os.makedirs(pathlib.Path(self.save_path).parent, exist_ok=True)

	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score - self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		"""Saves model when validation loss decrease."""
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.save_path)
		self.val_loss_min = val_loss



def testing(param_loc, data_loc, model_loc):

	parser = argparse.ArgumentParser(description="Running ner...")

	params,_ = parser.parse_known_args()
	params.__dict__ = pickle.load(open(param_loc, "rb"))
	#params.device = torch.device('cpu')
	params.bidir = True
	# load data
	df = pd.read_csv(data_loc, sep='\t')
	data_params = {'batch_size': params.bs,
				  'shuffle': False,
				  'num_workers': 2}

	dev_set = biodata(df, name='dev')
	testloader = DataLoader(dev_set, **data_params)

	# load model
	model = charner(params).to(device=params.device)
	model.load_state_dict(torch.load(model_loc,map_location=params.device))
	loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
	
	# run model
	# save result
	outputs = []
	test_loss, test_steps = 0,0
	model.eval()
	with torch.no_grad():
		for _, data in tqdm(enumerate(testloader)):
			
			ids = data['ids'].to(params.device, dtype=torch.long)
			tar = data['targets'].to(params.device)
			d_output = model(ids).squeeze(-1)
			outputs.append(d_output.cpu().detach().numpy())
			d_loss = loss_function(d_output, tar)
			test_loss += d_loss.item()
			test_steps += 1

	ooo = torch.from_numpy(np.vstack(outputs)>1).float()
	create_pred_file(df, ooo, name=params.model_id, save_dir=params.save_dir)
	print('Prediction saved!!')

def recreate(ss:list):
	if len(ss) == 1:
		nl = [a for a in ss[0].__interval__()]
	elif len(ss)>1:
		nl = [ a for s in ss for a in s.__interval__()]
	else:
		return ('-','-')
	return (min(nl), max(nl))

class interval:
	def __init__(self, start, end):
		self.s = start
		self.e = end
	
	def __print__(self):
		print(self.s, self.e)
	
	def __show__(self):
		return self.s, self.e
		


if __name__ == '__main__':


	if len(sys.argv) != 4:
		log.error("Invalid input parameters. Format: \
				  \n python utils.py [param_loc] [data_loc] [model_loc]")
		sys.exit(0)


	[_, param_loc, data_loc, model_loc] = sys.argv

	testing(param_loc, data_loc, model_loc)




