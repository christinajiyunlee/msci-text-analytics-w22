import sys
import gensim.models
from gensim.test.utils import datapath
from gensim.models.word2vec import PathLineSentences
from gensim import utils

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please run the script in the correct format. For ex: python main.py ../../textstyletransferdata/sentiment")
        exit()

    path = sys.argv[1]
    files = ['pos.txt', 'neg.txt']

    sentences = PathLineSentences(path)
    model = gensim.models.Word2Vec(sentences=sentences)
    model.save("word2vec.model")
    print("Model is saved!")