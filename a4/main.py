"""
This script  keras to train a fully-connected feed-forward neural network classifier to classify documents in the
Amazon corpus into positive and negative classes 3 different models (ReLU, Sigmoid, Tanh) on the training data from A1.
Command line argument:
    arg1: Path to the split .txt files from A1. There will be one word per line.
"""

import sys
import tensorflow as tf
import numpy as np
import os
import pandas as pd

from gensim.models import Word2Vec
from tensorflow.keras import layers, activations, regularizers

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, Embedding, Flatten, Dropout, BatchNormalization, Activation, Bidirectional
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Hyperparameters that can be tuned
MAX_SENT_LEN = None
MAX_VOCAB_SIZE = 20000
LSTM_DIM = 128
EMBEDDING_DIM = 100
BATCH_SIZE = 32
N_EPOCHS = 10

def train_model(type):

    # PREPROCESSING DATA
    df_train, df_val, df_test = load_data(path)
    # Pre-processing involves removal of puctuations and converting text to lower case
    word_seq = [text_to_word_sequence(sent) for sent in df_train['sentence']]
    print('90th Percentile Sentence Length:', np.percentile([len(seq) for seq in word_seq], 90))

    tokenizer = Tokenizer(num_words=len(df_train))
    tokenizer.fit_on_texts([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
    print("Number of words in vocabulary:", len(tokenizer.word_index))

    # Convert the sequence of words to sequnce of indices
    X = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
    X = pad_sequences(X, maxlen=MAX_SENT_LEN, padding='post', truncating='post')
    y = df_train['polarity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.1)

    # BUILD MODEL
    model = Sequential()
    print("\n>> Model is created.")

    # 1. Input layer of the word2vec embeddings you prepared in Assignment 3.
    embeddings, embeddings_matrix = load_word2vec(tokenizer)
    model = add_input_layer(embeddings, embeddings_matrix, tokenizer, model)

    # 2. Hidden activation layer: {ReLU, sigmoid, tanh}
    TYPE = type
    model = add_activation_layer(TYPE, model)
    print("\n>> "+ TYPE.title() +" Layer added.")

    model.add(Flatten())

    # 3. Final layer with softmax activation function.
    model.add(layers.Activation('softmax'))
    print("\n>> Softmax Layer added.")

    # 4. Use cross-entropy as the loss function.
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("\n>> Cross-Entropy Loss Layer added.")

    # 6. Add dropout. Try a few different dropout rates.
    model.add(Dropout(0.2))
    print("\n>> Dropout Layer added.\n")

    # CHECK MODEL
    model.summary()

    # TRAIN
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=N_EPOCHS,
              validation_data=(X_test, y_test))
    print("\n>> Finished training model.")

    # CALCULATE ACCURACY ON TEST SET
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Accuracy on Test Set = {0:4.3f}".format(acc))


""" Takes the path of the split data from A1 as input and outputs the loaded test, training and validations sets """
def load_data(path):
    print("\n>> Loading data from "+path)
    files = ["train", "val", "test"] # with stopwords
    files_ns = ["train_ns", "val_ns", "test_ns"] # without stopwords
    data = []
    for file in files:
        with open(path + "_fixed/" + file + ".csv", 'rb') as f:
                data.append(f.read().decode("utf-8").splitlines())

    #    Load tokenized training, testing, validating sets
    d_train, d_val, d_test = data #array of arrays of tokenized words

    #    Load labels
    labels = []
    with open(path + "_fixed/labels.csv", 'rb') as f:
        labels = (f.read().decode("utf-8").splitlines())
    y_train = labels[:len(d_train)]
    y_val = labels[len(d_train):len(d_train)+len(d_val)]
    y_test = labels[len(d_train)+len(d_val):]

    df_train = create_x_y("train", d_train, y_train)
    df_val = create_x_y("val", d_val, y_val)
    df_test = create_x_y("test", d_test, y_test)

    return df_train, df_val, df_test

""" Takes an array of arrays of tokens that was loaded and outputs clean X, y arrays with word_indexes """
def create_x_y(type, tokens, labels):
    df = pd.DataFrame(columns=['sentence', 'polarity'])

    word_seq = []
    y = []
    for seq in tokens:
        sentence = " ".join([ str(i) for i in seq[2:-2].split('\', \'') ])
        word_seq.append(sentence)

    df['sentence'] = word_seq
    df['polarity'] = labels
    df = df.sample(frac=1, random_state=10) # Shuffle the rows
    df.reset_index(inplace=True, drop=True)
    print("\n>> "+ type.title() +" Data is loaded.")

    return df


#
""" Loads the word2vec model from A3 and outputs the embeddings_vector and embeddings """
def load_word2vec(tokenizer):
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../a3/word2vec.model')
    embeddings = Word2Vec.load(filename).wv
    print("The word2vec embedding is of size: " + str(embeddings.vector_size))

    # Create an embedding matrix containing only the word's in our vocabulary
    # If the word does not have a pre-trained embedding, then randomly initialize the embedding
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, EMBEDDING_DIM)) # +1 is because the matrix indices start with 0

    for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = embeddings[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    return embeddings, embeddings_matrix

""" Adds the input layer to the model """
def add_input_layer(embeddings, embeddings_matrix, tokenizer, model):
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1,
                        output_dim=embeddings.vector_size,
                        weights = [embeddings_matrix],
                        trainable=False,
                        name='word_embedding_layer',
                        input_length=MAX_SENT_LEN,
                        mask_zero=True))
    print("\n>> Embedding Layer added.")
    return model

""" Adds the specified activation layer to the model """
def add_activation_layer(type, model):
    if type == "relu":
        model.add(layers.Dense(128, kernel_regularizer=l2(0.01), activation=activations.relu, name="relu_layer"))
    elif type == "sigmoid":
        model.add(Dense(128, kernel_regularizer=l2(0.01), activation='sigmoid', name='sigmoid_layer'))
    elif type == "tanh":
        model.add(layers.Dense(128, kernel_regularizer=l2(0.01), activation=activations.relu, name="tanh"))
    else:
        print("*** Type is not one of: relu, sigmoid, tanh. Exiting.***")
        exit()
    return model


if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please run the script in the correct format. For ex: python a4/main.py a1/data")
        exit()

    path = sys.argv[1]
    model = train_model("relu")
    model.save("nn_relu.model")

    model = train_model("sigmoid")
    model.save("nn_sigmoid.model")

    model = train_model("tanh")
    model.save("nn_tanh.model")
