import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn

from data import *
from model import *
from utils import *


loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()

def train(model, trainloader, epoch, optimizer):
	tr_loss, tr_steps = 0,0
	n_correct = 0
	
	model.train()
	for _, data in tqdm(enumerate(trainloader)):
		optimizer.zero_grad()
		ids = data['ids']#.to(device, dtype=torch.long)
		tar = data['targets']#.to(device, dtype=torch.long)
		output = model(ids).squeeze(-1)
		loss = loss_function(output, tar)
		tr_loss += loss.item()
		tr_steps += 1
		
		if _ % 50 == 0:
			print(f' Training loss per 50 step : {tr_loss/ tr_steps}')
			
		loss.backward()
		optimizer.step()
		
def eval(model, testloader):
	test_loss, test_steps = 0,0
	n_correct = 0
	
	model.eval()
	with torch.no_grad():
		for _, data in tqdm(enumerate(testloader)):
			
			ids = data['ids']#.to(device, dtype=torch.long)
			tar = data['targets']#.to(device, dtype=torch.long)
			output = model(ids).squeeze(-1)
			loss = loss_function(output, tar)
			test_loss += loss.item()
			test_steps += 1

		
		print(f'Testing loss: {test_loss/ test_steps}')



def main(params):

	# Load dataset
	train_df = pd.read_csv('~/bio-challenge/data/BioCreative_TrainTask3.0.tsv', sep='\t')
	dev_df = pd.read_csv('~/bio-challenge/data/BioCreative_ValTask3.tsv', sep='\t')

	train_df = pd.read_csv('~/bio-challenge/data/SMM4H18_train_modified.csv', sep='\t')
	train_df.head()


	# positive samples
	tpdf = train_df.loc[train_df['start'] != '-']
	tndf = train_df.loc[train_df['start'] == '-']
	dpdf = dev_df.loc[dev_df['start'] != '-']
	dndf = dev_df.loc[dev_df['start'] == '-']

	# traindf - subsampled tdf
	tdf = pd.concat([tpdf,tndf.iloc[:877]])
	tdf = tdf.sample(frac=1)
	# devdf - subsampled ddf
	ddf = pd.concat([dpdf,dndf.iloc[:396]])
	ddf = ddf.sample(frac=1)

	# load dataloader
	train_set = biodata(tdf, name='train')
	dev_set = biodata(ddf, vocab=train_set.vocab, name='dev')

	train_params = {'batch_size': params.bs,
			   'shuffle':True,
			   'num_workers': 2}
	dev_params = {'batch_size': params.bs,
				  'shuffle': False,
				  'num_workers': 2}

	trainloader = DataLoader(train_set, **train_params)
	devloader = DataLoader(dev_set, **dev_params)

	model = nermodel(params)
	optimizer = torch.optim.Adam(params =model.parameters(), lr=params.lr)
	EPOCHS = 2

	for e in range(EPOCHS):
		train(model, trainloader, e, optimizer)
		if e % params.test_every == 0:
			eval(model, devloader)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Running ner...")
	parser.add_argument('--device', default = torch.device('cpu'), 
						help='cpu or gpu')
	parser.add_argument('--hs', default=768, type=int,
						help='Hidden layer size')
	parser.add_argument('--bs', default=32, 
						help='batch size')
	parser.add_argument('--nl', default=2, 
						help='Number of layers')
	parser.add_argument('--bidir', default=1,
					   help='bi-directional')
	parser.add_argument('--inplen', default=50,
					   help='sequence/sentence length')
	parser.add_argument('--inpsize', default=768,
					   help='embedding size')
	parser.add_argument('--vocabsize', default=768,
					   help='vocab size')
	parser.add_argument('--lr', default= 0.001, type=float,
						help="Learning rate of loss optimization")
	parser.add_argument('--test_every', default=1,
					   help='Testing after steps')


	params,_ = parser.parse_known_args()


	main(params)