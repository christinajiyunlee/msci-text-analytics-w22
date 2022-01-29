## Assignment 1

### About the Code
This driver script `main.py` does the following tasks:
1. Tokenize the corpus
2. Remove the following special characters: !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n
3. Create two versions of your dataset: (1) with stopwords and (2) without stopwords.
Stopword lists are available online.
4. Randomly split your data into training (80%), validation (10%) and test (10%) sets.

###Instructions to Run Code
1. Navigate to the project directory `msci-text-analytics-w22`
2. Run the driver script `python a1/main.py '../textstyletransferdata/sentiment'`
3. Check these 8 expected files have output under `a1/data` folder
   - out.csv: tokenized sentences w/ stopwords 
   - train.csv: training set w/ stopwords 
   - val.csv: validation set w/ stopwords 
   - test.csv: test set w/ stopwords 
   - out_ns.csv: tokenized sentences w/o stopwords
   - train_ns.csv: training set w/o stopwords 
   - val_ns.csv: validation set w/o stopwords
   - test_ns.csv: test set w/o stopwords
