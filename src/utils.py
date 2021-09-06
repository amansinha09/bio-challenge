import numpy as np
import pandas as pd



def create_pred_file(ddf, output, name):
	predques = ddf.iloc[:, 0:4].copy()

	spans = []
	for t in ooo:
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
	pred.to_csv(f'pred_{name}.csv')


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