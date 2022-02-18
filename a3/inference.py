"""
This script generates the top-20 most similar words for a given word.
Command line argument - arg1: Path to a .txt file, which contains some words compiled for evaluation. There will be one word per line.
"""
import sys
import re
from gensim.models import Word2Vec

def tokenize(word):
    patterns = [r'\w+'] # remove special characters like !"#$%&()*+/:;<=>@[\]^`{|}~\t\n

    for p in patterns:
        word = re.findall(p, word)
    return word[0]

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please run the script in the correct format. For ex: python inference.py test.txt")
        exit()

    path = sys.argv[1]

    # Load word2vec model
    model = Word2Vec.load("word2vec.model")

    # Read in txt file
    input_words = []
    with open(path, 'rb') as f:
        input_words = f.read().decode("utf-8").splitlines()

    # Read one word at a time
    for word in input_words:
        if len(word.split(" "))>1:
            print("Please input only one word per line in your .txt file")
            exit()

        print("\n==============================")
        print("Input word is: "+ tokenize(word))
        print("\nTop 20 similar words are:")
        output_words = model.wv.most_similar(positive=[word],topn=20)
        for w in output_words:
            print(w[0] +": " + str(w[1]))
