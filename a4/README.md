## Assignment 4

###Report Content

| Activation Function  |   Accuracy Result     |
|:----------:|:-------------:|
| ReLU |  ___ |
| Sigmoid |    ___   |
| Tanh | ___ | 

###Instructions to Run Code
1. Navigate to the project directory `msci-text-analytics-w22`
2. Run the script `python a2/main.py a1/data_fixed TYPE_OF_CLASSIFIER`, where `TYPE_OF_CLASSIFIER` is one of:
    - mnb_uni_ns
    - mnb_bi_ns
    - mnb_uni_bi_ns
    - mnb_uni
    - mnb_bi
    - mnb_uni_bi
    
### About the Code
This script `inference.py` does the following tasks:
1. Trains a Multinomial Naïve Bayes (MNB) classifier to classify the documents in the Amazon 
   corpus into positive and negative classes.
2. Produces the following files:
    - mnb_uni.pkl: Classifier for unigrams w/ stopwords 
    - mnb_bi.pkl: Classifier for bigrams w/stopwords 
    - mnb_uni_bi.pkl: Classifier for unigrams+bigrams w/ stopwords 
    - mnb_uni_ns.pkl: Classifier for unigrams w/o stopwords 
    - mnb_bi_ns.pkl: Classifier for bigrams w/o stopwords 
    - mnb_uni_bi_ns.pkl: Classifier for unigrams+bigrams w/o stopwords 
3. The MNB model was tuned by experimenting with several alpha values with 1 of the 6 classifier type variations to see which produces the best accuracy. The alpha value with best accuracy was 0.5.
4. Assignment 1 was modified to ensure input values matched expected formats for assignment 2. The new code can be found under `a1/data_fixed` and `main_fixed.py`
