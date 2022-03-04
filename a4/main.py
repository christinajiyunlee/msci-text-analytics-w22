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

from gensim.models import Word2Vec
from tensorflow.keras import layers, activations, regularizers

from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, Embedding, Dropout, BatchNormalization, Activation, Bidirectional

# Hyperparameters that can be tuned
MAX_SENT_LEN = 30
MAX_VOCAB_SIZE = 20000
LSTM_DIM = 128
EMBEDDING_DIM = 100
BATCH_SIZE = 32
N_EPOCHS = 10

""" Takes the path of the split data from A1 as input and outputs the loaded test, training and validations sets """
def load_data(path):
    print("\n>> Loading data from "+path)
    files = ["train", "test", "val"] # with stopwords
    files_ns = ["train_ns", "test_ns", "val_ns"] # without stopwords
    data = []
    for file in files:
        with open(path + "/" + file + ".csv", 'rb') as f:
            data.append(f.read().decode("utf-8").splitlines())

    #    Load tokenized training, testing, validating sets
    d_train, d_test, d_val = data #array of arrays of tokenized words

    X_train, y_train, word_index = create_x_y("train", d_train)
    X_test, y_test, _= create_x_y("test", d_test)
    X_val, y_val, _ = create_x_y("val", d_val)

    return X_train, y_train, X_val, y_val, X_test, y_test, word_index

""" Takes an array of arrays of tokens that was loaded and outputs clean X, y arrays with word_indexes """
def create_x_y(type, tokens):
    word_seq = []
    y = []
    for seq in tokens:
        s = seq.split("\"")
        sentence = s[0]+s[1]
        label = s[2].split(",")[2]
        word_seq.append([ str(i) for i in sentence[2:-2].split('\', \'') ])
        y.append(label) # Load labels as y variable
    print("\n>> "+ type.title() +" Data is loaded.")

    #    Create word index
    word_index = {}
    index = 1
    X = []
    for seq in word_seq:
        seq_i = []
        for token in seq:
            if token not in word_index:
                word_index[token] = index
                index += 1
            # Convert the sequence of words to sequnce of indices
            seq_i.append(word_index[token])
        X.append(seq_i)
    print("Number of words in "+ type +" index:", len(word_index))

    if type == "train":
        index = word_index
    else:
        index = None
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, index

""" Loads the word2vec model from A3 and outputs the embeddings_vector and embeddings """
def load_word2vec():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../a3/word2vec.model')
    embeddings = Word2Vec.load(filename).wv
    print("The word2vec embedding is of size: " + str(embeddings.vector_size))

    #    Create an embedding matrix using only the tokens that exist in training set
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_index)+1, embeddings.vector_size))
    for word, i in word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = embeddings[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    return embeddings, embeddings_matrix

""" Adds the input layer to the model """
def add_input_layer(embeddings, embeddings_matrix, model):
    model.add(Embedding(input_dim=len(word_index)+1,
                        output_dim=embeddings.vector_size,
                        weights = [embeddings_matrix], trainable=False, name='word_embedding_layer',
                        mask_zero=True))
    print("\n>> Embedding Layer added.")
    return model

""" Adds the specified activation layer to the model """
def add_activation_layer(type, model):
    if type == "relu":
        model.add(layers.Dense(128, activation=activations.relu, name="relu_layer"))
    elif type == "sigmoid":
        model.add(Dense(128, activation='sigmoid', name='sigmoid_layer'))
    elif type == "tanh":
        model.add(layers.Dense(128, activation=activations.relu, name="tanh"))
    else:
        print("*** Type is not one of: relu, sigmoid, tanh. Exiting.***")
        exit()
    return model


if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Please run the script in the correct format. For ex: python a4/main.py a1/data")
        exit()

    path = sys.argv[1]

    # PREPROCESSING DATA
    X_train, y_train, X_val, y_val, X_test, y_test, word_index = load_data(path)

    # BUILD MODEL
    model = Sequential()
    print("\n>> Model is created.")

    # 1. Input layer of the word2vec embeddings you prepared in Assignment 3.
    embeddings, embeddings_matrix = load_word2vec()
    model = add_input_layer(embeddings, embeddings_matrix, model)

    # 2. Hidden activation layer: {ReLU, sigmoid, tanh}
    TYPE = "sigmoid" #TODO: repeat for all 3 types by creating 3 different models
    model = add_activation_layer(TYPE, model)
    print("\n>> "+ TYPE.title() +" Layer added.")

    # 3. Final layer with softmax activation function.
    model.add(layers.Activation('softmax'))
    print("\n>> Softmax Layer added.")

    # 4. Use cross-entropy as the loss function.
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print("\n>> Cross-Entropy Loss Layer added.")

    # 5. Add L2-norm regularization.
    model.add(Dense(128, kernel_regularizer=l2(0.01)))
    print("\n>> L2-Norm Regularization Layer added.")

    # 6. Add dropout. Try a few different dropout rates.
    model.add(Dropout(0.2)) # TODO: Test different dropout rates
    print("\n>> Dropout Layer added.\n")

    # CHECK MODEL
    model.summary()

    # TRAIN
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=N_EPOCHS,
              validation_data=(X_test, y_test)) #TODO: Fix the datatype issue here.
    print("\n>> Finished training model.")

    # CALCULATE ACCURACY ON TEST SET
    score, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Accuracy on Test Set = {0:4.3f}".format(acc))

    # EXPORT MODEL FILES
    model.save("nn_sigmoid.model") # TODO: do for all 3 types of models ("nn_relu.model", "nn_sigmoid.model", "nn_tanh.model")
    # TODO: Try with stemmed & unstemmed see which one gets better accuracy
