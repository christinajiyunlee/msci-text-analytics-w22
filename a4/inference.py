"""
This script classifies a set of test lines as positive or negative using one of the 3 trained models from the main.py script
Command line arguments:
    arg1: Path to a .txt file, which contains some words compiled for evaluation. There will be one word per line.
    arg2: Type of classifier to use. {relu, sigmoid, tanh}

Make Predictions
"""

import sys

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Please run the script in the correct format. For ex: python a4/inference.py a4/test.txt relu")
        exit()

    path = sys.argv[1]
    classifier = sys.argv[2]

    filename = 'nn_'+ classifier+'.model'
    model =keras.models.load_model(filename)

    model.predict(X_test[:5])

    to_be_predicted = []
    with open(path, 'rb') as f:
        to_be_predicted.append(f.read().decode("utf-8").splitlines())

    word_seq = [text_to_word_sequence(sent) for sent in to_be_predicted]
    X_predict = tokenizer.texts_to_sequences([' '.join(seq[:MAX_SENT_LEN]) for seq in word_seq])
    X_predict = pad_sequences(X_predict, maxlen=MAX_SENT_LEN, padding='post', truncating='post')

    print(X_predict)



