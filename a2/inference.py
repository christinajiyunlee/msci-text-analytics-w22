# TODO: create inference.py file that takes a testing.txt file in a2/data and outputs in the command line whether 1: positive, 0: negative
import sys
import re
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from main import *

def tokenize(corpus):
    tokens = []
    patterns = [r'\w+'] # remove special characters like !"#$%&()*+/:;<=>@[\]^`{|}~\t\n
    for sentence in corpus:
        words = []
        s = (sentence).split(' ')
        for w in range(len(s)): # separate by spaces into 1d array
            for p in patterns:
                words.extend(re.findall(p, s[w]))
        tokens.append(words)

    return tokens


def remove_stopwords(tokens):
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for k in tokens:
        sentence = tokens[k]
        for word in sentence:
            if word.lower() in stopwords:
                sentence.remove(word)
    return tokens

def predict(x, clf, count_vect, tfidf_transformer):
    """
    Takes the testing/validation data and trained models to output predictions
    :param x: List of arrays that are tokenized sentences
    :param clf: MNB model
    :param count_vect: count vectorizer
    :param tfidf_transformer: tfidf transformer
    :return: Predictions for all x
    """
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)

    return preds.tolist()

def main(txt_dir, classifier_type):

    ns = False
    ngram = ""
    ctype = classifier_type.split("_")
    if "ns" in ctype:
        ns = True
    if "uni" in ctype and "bi" in ctype:
        ngram = "multi"
    elif "uni" in ctype:
        ngram = "uni"
    elif "bi" in ctype:
        ngram = "bi"
    print("\nStopwords removed: "+str(ns))
    print("Type of word-gramming: " + ngram + "\n")

    # Run evaluation on input txt file
    with open(txt_dir+'/input.txt', 'r') as f:
        lines = f.readlines()

        # Tokenize lines into x,y format
        x = [' '.join(line) for line in tokenize(lines)]

        # Import the correct model from pkl file
        with open('a2/data/'+classifier_type+'.pkl', 'rb') as f:
            clf = pickle.load(f)

        # Import countvectorizer
        print('Calling CountVectorizer') # Counts the number of times each token, assigned a unique value, occurs in the corpus
        with open('a2/data/countvec_'+classifier_type+'.pkl', 'rb') as f:
            count_vect = pickle.load(f)

        # Import tfidf
        print('Building Tf-idf vectors') # Identifies the most significant words
        with open('a2/data/tfidf_'+classifier_type+'.pkl', 'rb') as f:
            tfidf_transformer = pickle.load(f)

        predictions = predict(x, clf, count_vect, tfidf_transformer)

        # Write predictions to output.txt file
        with open('a2/data/output.txt', mode='w') as f:
            for p in predictions:
                f.write(str(p))

        # Print predictions to terminal
        for i in range(len(lines)):
            print('\nSENTENCE: %s' % lines[i].rstrip())
            print('POSITIVE(1) OR NEGATIVE(0): %s' % predictions[i])

        return predictions


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please run the script in this format: python a2/inference.py PATH_TO_INPUT_TXT_FILE TYPE_OF_CLASSIFIER")
        exit()

    txt_dir = sys.argv[1]
    classifier_type = sys.argv[2]

    main(txt_dir, classifier_type)
