from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	
	def __init__(self, model, tokenizer):
		super(Encoder,self).__init__()
		self.model = model
		self.tokenizer = tokenizer

	@abstractmethod
	def encode(self, pad=None):
		pass

	def forward(self, inputs):
		pass

	@abstractmethod
	def collate_fn(self, inputs):
		pass


class BertEncoder(Encoder):
	def __init__(self, model, tokenizer):
		super(BertEncoder, self).__init__(model, tokenizer)

	def encode(self, seq, pad=50):

		tok2span = {}
		tok_ids = []
		span = []
		start, end = 0,0

		# cls token
		#tok_ids.append(self.tokenizer.cls_token_id)
		#end += 1
		#tok2span['CLS'] = [start, end]
		#span.append([start, end])
		#start = end

		# iterate over sequence and encode on token ids
		for tok_id, tok in enumerate(seq):
			tok_encoding = self.tokenizer.encode(tok, add_special_tokens=False)
			end += len(tok_encoding)
			tok2span[tok] = [start, end]
			span.append([start, end])
			start = end
			tok_ids.extend(tok_encoding)

		# add sep token
		#tok_ids.append(self.tokenizer.sep_token_id)
		#end += 1
		#tok2span['SEP'] = [start, end]
		#span.append([start, end])

		# token ids to tensor
		tok_ids = torch.tensor(tok_ids)

		# pad
		if pad and len(tok_ids) < pad:
			tok_ids = F.pad(tok_ids, (0,pad-len(tok_ids)),value=self.tokenizer.pad_token_id)

		# attention mask on pad tokens
		att_mask = torch.ne(tok_ids, self.tokenizer.pad_token_id)
		span.extend([[end,end] for x in range(len(tok_ids)-len(span))])

		# span to tensor
		span = torch.tensor(span)
		#print('==>',tok_ids.shape, att_mask.shape, span.shape)
		#if tok_ids.size(0)>pad: print(tok_ids, span)
		return tok_ids[:pad], att_mask[:pad], span[:pad]

	def collate_fn(self, inputs):

		tok_ids, att_mask, spans =  [torch.stack(x) for x in list(zip(*inputs))]

		return (tok_ids, att_mask, spans)


	def forward(self, inputs):
		"""Run transformer model on inputs. Average bpes per token and remove cls and sep vectors"""

		tok_ids, att_mask, span = inputs
		#print('tids:', tok_ids.shape, )
		#print('***********',self.model.device)#tok_ids.shape, att_mask.shape, span.shape)
		with torch.autograd.no_grad():

			outputs  = self.model(tok_ids, attention_mask=att_mask, output_hidden_states=True) # mask is used for pad tokens
			#if encoder_type == 'linear':
			output = outputs[0]
			#print('os',output.shape)
			#else:
			#	output = torch.sum(torch.Tensor([outputs['hidden_states'][-i].detach().cpu().numpy() for i in range(1,5)]), dim=0)
			#print(output.shape)

			# compute number of bpe per token
			first_bpe = span[:,:,0] # first bpe indice
			last_bpe = span[:,:,1] # last bpe indice
			n_bpe = last_bpe-first_bpe # number of bpe by token = first - last bpe from span
			#print(first_bpe.shape, last_bpe.shape, n_bpe.shape)
			# mask pad tokens
			mask = n_bpe.ne(0)
			n_bpe = n_bpe[mask] # get only actual token bpe
			#print('new shape nbpe:',n_bpe.shape)
			# compute mean : sum up corresponding bpe then divide by number of bpe
			indices = torch.arange(n_bpe.size(0), device=self.model.device).repeat_interleave(n_bpe) # indices for index_add
			#print(indices.shape, span.shape, len(indices), len(output[att_mask]))
			#print(indices, att_mask)
			k = min(len(indices),len(output[att_mask]))
			average_vectors = torch.zeros(n_bpe.size(0), output.size(2), device=self.model.device) # starts from zeros vector
			average_vectors.index_add_(0, indices[:k], output[att_mask].to(device=self.model.device)) # sum of bpe based in indices
			average_vectors.div_(n_bpe.view(n_bpe.size(0),1)) # divide by number of bpe

			output_ = torch.zeros_like(output, device=self.model.device) # new output vector to match outputsize
			output_[mask] = average_vectors

			output = output_[:,:,:] # get rid of cls and sep

			return output
