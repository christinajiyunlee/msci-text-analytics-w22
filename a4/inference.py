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
        print("Please run the script in the correct format. For ex: python a4/inference.py a4/test.txt ")
        exit()

    path = sys.argv[1]
    classifier = sys.argv[2]