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