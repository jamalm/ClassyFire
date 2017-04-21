# import gaussian naive bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
# import numpy as np
import pandas as pd


def main():
    train_path = r'.\data\trainingset.txt'
    test_path = r'.\data\queries.txt'

    # read in training set
    train = pd.read_csv(train_path, header=None)
    # read in testing set
    test = pd.read_csv(test_path, header=None)

    (train_predictor, train_target) = setup_train(train)
    #print train_predictor
    #print "-----------------------------------------"
    #print train_target

    # model for continuous data
    gnb = GaussianNB()
    # gnb.fit(train_predictor, train_target)


# assign predictor and target variable
def setup_train(train):

    # cut off id var
    train.drop(train.columns[0], axis=1, inplace=True)

    # separate output variable as targets by popping last indexed column
    targets = train.pop(train.columns[-1])

    targets = tokenize(targets)
    train = tokenize(train)

    # return the predictor set and the target set for training
    return train, targets

def tokenize(df):

    # see http://stackoverflow.com/questions/28016752/sklearn-trying-to-convert-string-list-to-floats
    S = set(df) # collect unique label names
    D = dict(zip(S, range(len(S)))) # assign each string an integer, put it in a dict
    Y = [D[frame] for frame in df] # store class labels as ints
    print df
    return pd.DataFrame(Y)

if __name__ == '__main__':
    main()
