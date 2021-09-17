import os
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from data import *
from model import *
from utils import *


def main(params):

	if params.save_params:
		with open(f'/home/amsinha/bio-challenge/.logs/params_{params.model_id}.pkl','wb') as fp:
			pickle.dump(params.__dict__, fp)
	#return
	# Load dataset
	train_df0 = pd.read_csv('~/bio-challenge/data/BioCreative_TrainTask3.0.tsv', sep='\t')
	#test_df = pd.read_csv('~/bio-challenge/data/BioCreative_ValTask3.tsv', sep='\t')
	test_df = pd.read_csv('~/bio-challenge/data/BioCreative_TEST_Task3_PARTICIPANTS.tsv',sep='\t')
	train_df1 = pd.read_csv('~/bio-challenge/data/SMM4H18_train_modified.csv', sep='\t')
	train_df1 = train_df1.drop(columns=['Unnamed: 0'], axis=1)
	#train_df.head()

	# positive samples
	#tpdf = train_df.loc[train_df['start'] != '-']
	#tndf = train_df.loc[train_df['start'] == '-']
	#dpdf = dev_df.loc[dev_df['start'] != '-']
	#dndf = dev_df.loc[dev_df['start'] == '-']

	# traindf - subsampled tdf
	tdf = pd.concat([train_df0,train_df1], ignore_index=True)#, train_df1])
	tdf = tdf.sample(frac=1, random_state = 42)
	# devdf - subsampled ddf
	#ddf = pd.concat([dpdf,dndf])
	#ddf = ddf.sample(frac=1)
	#dpdf.to_csv(f'~/bio-challenge/ref/dpdf.csv', sep='\t')

	train_df, dev_df = train_test_split(tdf, test_size=0.3, random_state = 42)

	print(f"Number of train samples: {len(train_df)}")
	print(f"Number of dev samples: {len(dev_df)}")

	# load dataloader
	train_set = biodata2(train_df, name='train', level=params.level, use_bert=params.use_bert)
	dev_set = biodata2(dev_df, vocab=train_set.vocab, name='dev', max_len = train_set.max_len, level=params.level, use_bert=params.use_bert)
	test_set = biodata2(test_df, vocab=train_set.vocab, name='test', max_len= train_set.max_len, level=params.level, use_bert=params.use_bert)

	params.vocabsize = len(train_set.vocab)
	print(len(train_set.vocab))

	train_params = {'batch_size': params.bs,
			   'shuffle':True,
			   'num_workers': 2}
	dev_params = {'batch_size': params.bs,
				  'shuffle': False,
				  'num_workers': 2}
	test_params = dev_params

	trainloader = DataLoader(train_set, **train_params, drop_last=True)
	devloader = DataLoader(dev_set, **dev_params)
	testloader = DataLoader(test_set, **test_params)

	#for _,data in enumerate(trainloader):
	#	print(data['ids'].shape)
	#return

	#model = charner(params).to(device=params.device)
	model = bertner(params).to(device=params.device)

	loss_function = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params =model.parameters(), lr=params.lr)

	EPOCHS = params.epochs
	all_steps = 0

	writer = SummaryWriter(f'{params.save_dir}/ner_{params.model_id}')
	early_stopping = EarlyStopping(patience=5, verbose=True, save_path=f'{params.save_dir}/ner_{params.model_id}.pt')

	eloss = []
	for e in range(EPOCHS):
		tr_loss, tr_steps = 0,0

		model.train()
		for _, data in tqdm(enumerate(trainloader)):
			optimizer.zero_grad()
			ids = data['ids'].to(params.device, dtype=torch.long)
			tar = data['targets'].to(params.device)
			sp = data['spans'].to(params.device)
			attm = data['attn_mask'].to(params.device) 
			inp = (ids, attm, sp) if params.use_bert else ids
			output = model(inp).squeeze(-1)#;continue
			#output = model(ids).squeeze(-1)
			#print(output.shape, output[0], sp[0])
			loss = loss_function(output, tar)
			tr_loss += loss.item()
			tr_steps += 1
			all_steps += 1
			
			if _ % 50 == 0:
				print(f'Average training loss after {tr_steps} step, {e+1}/{EPOCHS} epoch : {(np.sum(eloss)+tr_loss)/ all_steps}')
				writer.add_scalar('Average Training loss ', (np.sum(eloss)+tr_loss)/ all_steps, all_steps)
			loss.backward()
			optimizer.step()

		eloss.append(tr_loss)

		if e % params.test_every == 0:
			dev_loss, dev_steps = 0,0
			outputs = []
			model.eval()
			with torch.no_grad():
				for _, data in tqdm(enumerate(devloader)):
					
					ids = data['ids'].to(params.device, dtype=torch.long)
					tar = data['targets'].to(params.device)
					sp = data['spans'].to(params.device)
					attm = data['attn_mask'].to(params.device)
					inp = (ids, attm, sp) if params.use_bert else ids
					d_output = model(inp).squeeze(-1)
					#d_output = model(ids).squeeze(-1)
					outputs.append(d_output.cpu().detach().numpy())
					d_loss = loss_function(d_output, tar)
					dev_loss += d_loss.item()
					dev_steps += 1

				print(f'\n------- Dev loss after {e+1}/{EPOCHS}: {dev_loss/ dev_steps}')
				writer.add_scalar('Average dev loss ', dev_loss/ dev_steps, all_steps)
				
			#if params.save_preds or (e == EPOCHS-1):
			#	ooo = torch.from_numpy(np.vstack(outputs)>1).float()
			#	create_pred_file(dev_df, ooo, name=params.model_id + f'_{e}', save_dir='~/bio-challenge/res/')
			#	print(f'{e}th epoch prediction saved!!')

		if params.stop_early:
			early_stopping(dev_loss, model)
			if early_stopping.early_stop:
				print('Early stopping')
				break 

	################ Testing ################
	outputs, sps = [], []
	test_loss, test_steps = 0,0
	model.eval()
	with torch.no_grad():
		for _, data in tqdm(enumerate(testloader)):
			
			ids = data['ids'].to(params.device, dtype=torch.long)
			tar = data['targets'].to(params.device)
			sp = data['spans'].to(params.device)
			attm = data['attn_mask'].to(params.device)
			inp = (ids, attm, sp) if params.use_bert else ids
			d_output = model(inp).squeeze(-1)
			#d_output = model(ids).squeeze(-1)
			sps.append(sp.cpu().detach().numpy())
			outputs.append(d_output.cpu().detach().numpy())
			d_loss = loss_function(d_output, tar)
			test_loss += d_loss.item()
			test_steps += 1

	print(f'\n------- Test loss : {test_loss/ test_steps}')
	if params.save_preds:
		sps = np.asarray(sps); sps = np.vstack(sps)
		ooo = torch.from_numpy(np.vstack(outputs)>1).float()
		create_pred_file(test_df, ooo,np.asarray(sps), name=params.model_id, save_dir=params.save_dir, level=params.level)
		print(f'Test prediction saved!!')



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
	parser.add_argument('--bs', default=32, type=int,
						help='batch size')
	parser.add_argument('--nl', default=2, type=int,
						help='Number of layers')
	parser.add_argument('--bidir', default=False, action="store_true",
					   help='bi-directional')
	parser.add_argument('--do_attn', default=False, action="store_true",
					help = 'apply attention')
	parser.add_argument('--natth', default=1, type=int,
					help='Number of attention heads')
	parser.add_argument('--level', default='char', choices= ['word', 'char'], required=True,
						help='Type of model')
	parser.add_argument('--inplen', default=50,
					   help='sequence/sentence length')
	parser.add_argument('--inpsize', default=768,
					   help='embedding size')
	parser.add_argument('--vocabsize', default=768,
					   help='vocab size')
	parser.add_argument('--lr', default= 0.001, type=float,
						help="Learning rate of loss optimization")
	parser.add_argument('--test_every', default=5, type=int,
					   help='Testing after steps')
	parser.add_argument('--save_dir', default='./.model/',
					   help='Model dir')
	parser.add_argument('--model_id', default=1,
					   help='model name identifier')
	parser.add_argument('--save_preds', action="store_true", 
						help='whether to save the preditions')
	parser.add_argument('--save_model', action="store_true", 
						help='whether to save the model')
	parser.add_argument('--stop_early', action="store_true", 
						help='whether to use early stopping')
	parser.add_argument('--save_params', action="store_true", 
						help='whether to save model params')
	parser.add_argument('--use_bert', action="store_true", 
						help='whether to use bert encoder')
	

	params,_ = parser.parse_known_args()


	main(params)
