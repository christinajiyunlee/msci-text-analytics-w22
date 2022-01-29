import sys
import re
import csv
import random


def tokenize(corpus):
    tokens = {}
    patterns = [r'\w+'] # remove special characters like !"#$%&()*+/:;<=>@[\]^`{|}~\t\n
    label = 0
    for sentence in corpus:
        words = []
        s = (sentence).split(' ')
        for w in range(len(s)): # separate by spaces into 1d array
            for p in patterns:
                words.extend(re.findall(p, s[w]))
        tokens[label] = words
        label+=1
    return tokens


def split_sets(tokens, train, val):
    # Random Sample Training and Test Data
    key_list = list(tokens.keys())
    training = int(len(key_list)*train) #80%
    validation = training+int(len(key_list)*val) #10%

    random.shuffle(key_list)

    training_dict = dict((key, tokens[key]) for key in key_list[:training])
    validating_dict = dict((key, tokens[key]) for key in key_list[training:validation])
    testing_dict = dict((key, tokens[key]) for key in key_list[validation:])
    return training_dict, validating_dict, testing_dict


def remove_stopwords(tokens):
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for k in tokens:
        sentence = tokens[k]
        for word in sentence:
            if word.lower() in stopwords:
                sentence.remove(word)
    return tokens


def export_file(file, tokens, label):
    with open('data/'+file, mode='w+', newline='') as f:
        token_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for sentence in tokens:
            token_writer.writerow([tokens[sentence], sentence, label])
    print('Finished exporting '+file)


"""
python Tokenize.py ../../textstyletransferdata/sentiment/
"""
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(
            f"ERROR: Not enough arguments provided. Please run in this format: python Tokenize.py PATH_TO_TXT_FILES")
    elif len(sys.argv) > 2:
        sys.exit(
            f"ERROR: Too many arguments provided. Please run in this format: python Tokenize.py PATH_TO_TXT_FILES")

    path = (sys.argv)[1]

    files = ['pos', 'neg']
    for file in files:
        with open(path + '/' + file + '.txt') as f:
            corpus = f.readlines()

            tokens = tokenize(corpus)

            # export: out.csv - tokenized sentences w/ stopwords
            export_file('out.csv', tokens, file)

            # split tokens randomly into training (80%), validation (10%) and test (10%)
            training_set, validating_set, testing_set = split_sets(tokens, 0.8, 0.1)

            # export 80%: train.csv
            export_file('train.csv', training_set, file)

            # export 10%: val.csv
            export_file('val.csv', validating_set, file)

            # export 10%: test.csv
            export_file('test.csv', testing_set, file)

            # Remove stopwords. Stopwords from: https://gist.github.com/sebleier/554280
            tokens = remove_stopwords(tokens)

            # export: out_ns.csv - tokenized sentences w/o stopwords
            export_file('out_ns.csv', tokens, file)

            # split tokens randomly into training (80%), validation (10%) and test (10%)
            training_set, validating_set, testing_set = split_sets(tokens, 0.8, 0.1)

            # export 80%: train_ns.csv
            export_file('train_ns.csv', training_set, file)

            # export 10%: val_ns.csv
            export_file('val_ns.csv', validating_set, file)

            # export 10%: test_ns.csv
            export_file('test_ns.csv', testing_set, file)
