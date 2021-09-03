import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data import *
from model import *
from utils import *


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
	ddf.to_csv('~/bio-challenge/ref/ddf.csv', sep='\t')

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

	model = charner(params).to(device=params.device)

	loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params =model.parameters(), lr=params.lr)

	EPOCHS = params.epochs
	all_steps = 0

	writer = SummaryWriter(f'{params.save_dir}/ner_baseline_{params.model_id}')

	eloss = []
	for e in range(EPOCHS):
		tr_loss, tr_steps = 0,0

		model.train()
		for _, data in tqdm(enumerate(trainloader)):
			optimizer.zero_grad()
			ids = data['ids'].to(params.device, dtype=torch.long)
			tar = data['targets'].to(params.device)
			output = model(ids).squeeze(-1)
			loss = loss_function(output, tar)
			tr_loss += loss.item()
			tr_steps += 1
			all_steps += 1
			
			if _ % 50 == 0:
				print(f' Average training loss after {tr_steps} step, {e+1}/{EPOCHS} epoch : {(np.sum(eloss)+tr_loss)/ all_steps}')
				writer.add_scalar('Average Training loss ', (np.sum(eloss)+tr_loss)/ all_steps, all_steps)
			loss.backward()
			optimizer.step()

		eloss.append(tr_loss)

		if e % params.test_every == 0:
			test_loss, test_steps = 0,0
			outputs = []
			model.eval()
			with torch.no_grad():
				for _, data in tqdm(enumerate(devloader)):
					
					ids = data['ids'].to(params.device, dtype=torch.long)
					tar = data['targets'].to(params.device)
					d_output = model(ids).squeeze(-1)
					outputs.append(d_output.detach().numpy())
					d_loss = loss_function(d_output, tar)
					test_loss += d_loss.item()
					test_steps += 1

				print(f'Testing loss after {e+1}/{EPOCHS}: {test_loss/ test_steps}')
				writer.add_scalar('Average dev loss ', test_loss/ test_steps, all_steps)
				
				if params.save_preds and (e == params.epochs):
					ooo = torch.from_numpy(np.vstack(outputs)>1).float()
					create_pred_file(ddf, ooo, name=params.model_id)


	if params.save_model : 
		torch.save(model.state_dict(), f"{params.save_dir}/model_{params.model_id}_e{e}.pth"); print('model saved !!')   



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Running ner...")
	parser.add_argument('--device', default = torch.device('cuda'), 
						help='cpu or gpu')
	parser.add_argument('--hs', default=768, type=int,
						help='Hidden layer size')
	parser.add_argument('--epochs', default=1, type=int, 
						help='Number of epochs')
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
	parser.add_argument('--save_dir', default='./.model/',
					   help='Model dir')
	parser.add_argument('--model_id', default=1,
					   help='model name identifier')
	parser.add_argument('--save_preds', default=False, 
						help='whether to save the preditions')
	parser.add_argument('--save_model', default=False, 
						help='whether to save the model')
	


	params,_ = parser.parse_known_args()


	main(params)
