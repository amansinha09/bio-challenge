# bio-challenge
Competition :  Biocreative VII [Official Website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/)

Team : Biobot

Track 3: [Automatic extraction of medication names in tweets](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-3/)

# TODOs

- [x] Build corpora from SMM4H18 trainset
- [x] Make evaluation code ready
- [x] Make src ready from notebook
- [x] add gpu running feature
- [x] Put it on grid
- [ ] Add inference code
- [ ] Run initial baseline char-rnn - only biotrainset
- [ ] Create word-rnn model 
- [ ] Use different/custom losses
- [ ] Think augmenation
- [ ] Add requirements.txt


## Instructions

### Getting Started 

Clone the repository : ``` git clone https://github.com/amansinha09/bio-challenge.git```
Create a virtual environment (venv) with ```requirements.txt``` using the commans below:

```
python3 -m venv .venv
. .venv/bin/activate
pip3 install --upgrade pip
cd bio-challenge
pip3 install -r requirements.txt
```

### Directory Structure
```
bio-challenge/
	|-- data/
	|-- task3/
		|-- src/
			|-- data.py
			[-- utils.py
			|-- evaluation.py
			|-- model.py
			|-- run.py
		|-- ref/
		|-- res/
		|-- .logs/
		|-- .model/
		|--run_exp.sh
	|-- README.md
```

### For running experiment


```
cd tasks/
mkdir ref/ res/ .model/ .logs/
cd .logs/
sh ../run_exp.sh
```

Change the parameters for custom experiments :

```
usage: run.py [-h] [--device DEVICE] [--hs HS] [--epochs EPOCHS] [--bs BS] [--nl NL] [--bidir BIDIR]
              [--inplen INPLEN] [--inpsize INPSIZE] [--vocabsize VOCABSIZE] [--lr LR]
              [--test_every TEST_EVERY] [--save_model_dir SAVE_MODEL_DIR] [--model_id MODEL_ID]
              [--save_preds SAVE_PREDS]

Running ner...

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       cpu or gpu
  --hs HS               Hidden layer size
  --epochs EPOCHS       Number of epochs
  --bs BS               batch size
  --nl NL               Number of layers
  --bidir BIDIR         bi-directional
  --inplen INPLEN       sequence/sentence length
  --inpsize INPSIZE     embedding size
  --vocabsize VOCABSIZE
                        vocab size
  --lr LR               Learning rate of loss optimization
  --test_every TEST_EVERY
                        Testing after steps
  --save_model_dir SAVE_MODEL_DIR
                        Model dir
  --model_id MODEL_ID   model name identifier
  --save_preds SAVE_PREDS
                        whether to save the preditions

```

### For evalution

Add the prediction file into the res/ folder and test file in the ref/ folder, then run the evalution script: ``` python src/evaluation.py ref/ res/```.