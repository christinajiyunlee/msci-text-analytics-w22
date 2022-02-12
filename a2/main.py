import os
import sys
import pickle
from pprint import pprint
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def read_csv(data_path):
    """
    Takes a String path and reads contents of the file from file. Then returns it in a list format.
    :param data_path: String path that points to a file to be read
    :return: List of tokenized sentences
    """
    with open(data_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')).replace("'", "").replace("  ", " ") for line in data]


def load_data(data_dir, ns):
    """
    Loads the training, testing and validating datasets, either with/without stopwords
    :param data_dir: path to csv outputs from a1
    :param ns: True if no stopwords, False if includes stopwords
    :return: List of tokenized sentences in training, validating and testing sets
    """
    if ns:
        file_ext = "_ns"
    else:
        file_ext = ""

    x_train = read_csv(os.path.join(data_dir, 'train'+file_ext+'.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val'+file_ext+'.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test'+file_ext+'.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test

def train(x_train, y_train, ngram):
    """
    Takes x & y training data and trains a Multinomial Naive Bayes model.
    :param x_train: 80% of the entire neg & pos corpus
    :param y_train: labels for the 80% of the entire neg & pos corpus
    :return: MNB model
    """
    print('Calling CountVectorizer') # Counts the number of times each token, assigned a unique value, occurs in the corpus
    range = ()
    if ngram == "uni":
        range = (1, 1)
    elif ngram == "bi":
        range = (2, 2)
    elif ngram == "multi":
        range = (1, 2)
    count_vect = CountVectorizer(ngram_range=range)
    x_train_count = count_vect.fit_transform(x_train)

    print('Building Tf-idf vectors') # Identifies the most significant words
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_count)

    print('Training MNB') # Identifies the most significant words
    clf = MultinomialNB(alpha=0.5).fit(x_train_tfidf, y_train)
    return clf, count_vect, tfidf_transformer


def evaluate(x, y, clf, count_vect, tfidf_transformer):
    """
    Takes the testing/validation data and trained models to output predictions
    :param x: List of arrays that are tokenized sentences
    :param y: 1 or 0 labels for each tokenized sentence
    :param clf: MNB model
    :param count_vect: count vectorizer
    :param tfidf_transformer: tfidf transformer
    :return: Prediction score for model against each x,y set
    """
    x_count = count_vect.transform(x)
    x_tfidf = tfidf_transformer.transform(x_count)
    preds = clf.predict(x_tfidf)
    print(preds)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
    }


def main(classifier_type):
    """
    loads the dataset along with labels, trains a simple MNB classifier
    and returns validation and test scores in a dictionary
    """
    data_dir = "a1/data_fixed"
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

    # load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir, ns)

    # train
    clf, count_vect, tfidf_transformer = train(x_train, y_train, ngram)

    with open('a2/data/'+classifier_type+'.pkl', 'wb') as f:
        pickle.dump(clf, f)

    with open('a2/data/countvec_'+classifier_type+'.pkl', 'wb') as f:
        pickle.dump(count_vect, f)

    with open('a2/data/tfidf_'+classifier_type+'.pkl', 'wb') as f:
        pickle.dump(tfidf_transformer, f)

    scores = {}
    # validate
    print('Validating')
    scores['val'] = evaluate(x_val, y_val, clf, count_vect, tfidf_transformer)
    # test
    print('Testing')
    scores['test'] = evaluate(x_test, y_test, clf, count_vect, tfidf_transformer)

    return scores


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Please run the script in this format: python a2/main.py")
        exit()

    classifier_types = ["mnb_uni_ns", "mnb_bi_ns", "mnb_uni_bi_ns", "mnb_uni", "mnb_bi", "mnb_uni_bi"]
    for type in classifier_types:
        print(main(type))