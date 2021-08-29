# bio-challenge
Competition :  Biocreative VII [Official Website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/)

Team : Biobot

Track 3: [Automatic extraction of medication names in tweets](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-3/)


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

### For evalution

Create two folder in home directory : ```ref/``` and ```res/```. Add the prediction file into the res/ folder and test file in the ref/ folder, then run the evalution script: ``` python src/evaluation.py ref/ res/```.




## TODOs

- [x] Build corpora from SMM4H18 trainset
- [x] Make evaluation code ready
- [x] Make src ready from notebook
- [ ] add gpu running feature
- [ ] Put it on grid
- [ ] Run initial baseline char-rnn - only biotrainset
- [ ] Create word-rnn model 
- [ ] Use different/custom losses
- [ ] Think augmenation
- [ ] Add requirements.txt
