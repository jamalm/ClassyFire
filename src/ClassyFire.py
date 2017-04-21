# import gaussian naive bayes model
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
import pandas as pd


def main():
    train_path = r'.\data\trainingset.txt'
    test_path = r'.\data\queries.txt'

    # read in training set
    traindata = pd.read_csv(train_path, header=None)
    # read in testing set
    test = pd.read_csv(test_path, header=None)

    # first train the model and return the fitted model
    gnb = train(traindata)

    # use that model to perform the prediction on the testing set of data

"""
    # (train_predictor, train_target) = setup_train(train)

    #print train_predictor
    #print "-----------------------------------------"
    #print train_target

    # model for continuous data
    gnb = GaussianNB()
    # gnb.fit(train_predictor, train_target)
"""


def train(df):
    # fit independent NB models and transform them to a uniform feature set
    # This method uses the class assignment probabilities as new features (see predict_proba function)
    (tdata, ttarget) = merge_model(setup_models(df))
    gnb = GaussianNB()
    gnb.fit(tdata, ttarget)
    test = gnb.predict_proba(tdata)
    print "Test predictions"
    print test
    return gnb

def setup_models(train, test=False):

    # get index lists for categorical and continuous data
    cat_train = train.iloc[:, list(train.select_dtypes(include=['object']).columns)]
    con_train = train.iloc[:, list(train.select_dtypes(include=['int64']).columns)]
    cat_train[0] = cat_train.index
    # cut off id var , if test is true, save it somehow
    if(test):
        ids = cat_train.pop(cat_train.columns[0])
    else:
        cat_train.drop(cat_train.columns[0], axis=1, inplace=True)
    # train.drop(train.columns[0], axis=1, inplace=True)
    #

    # reset data column headers
    cat_train.columns = list(range(len(cat_train.columns)))
    con_train.columns = list(range(len(con_train.columns)))

    # separate output variable as targets by popping last indexed column
    targets = cat_train.pop(cat_train.columns[-1])
    # convert series to dataframe for header
    targets = targets.to_frame()
    # set column header to 0 for tokenise function to work
    targets.columns = [0]

    #tokenise categorical features for multinomial NB algorihm
    targets = tokenize(targets)
    cat_train = tokenize(cat_train)

    #normalise the continuous data for the gaussian NB algorithm


    # return the predictor set and the target set for training
    return con_train, cat_train, targets

def merge_model((con_predictors, cat_predictors, targets)):
    targetsample = targets[0]
    #perform gaussian on continuous data
    gnb = GaussianNB()
    gnb.fit(con_predictors, targetsample)
    gauss = gnb.predict_proba(con_predictors)
    # perform multinomial on categorical data
    mnb = MultinomialNB()
    mnb.fit(cat_predictors, targetsample)
    multi = mnb.predict_proba(cat_predictors)
    print "Gaussian predictions: "
    print gauss
    print "Multinomial predictions: "
    print multi

    # concatenate resulting class probabilities
    predictions = np.hstack((multi, gauss))

    return predictions, targets

def tokenize(df):
    """
    # see http://stackoverflow.com/questions/28016752/sklearn-trying-to-convert-string-list-to-floats
    S = set(df) # collect unique label names
    D = dict(zip(S, range(len(S)))) # assign each string an integer, put it in a dict
    Y = [D[frame] for frame in df] # store class labels as ints
    print df
    return pd.DataFrame(Y)
    """
    for column in df:
        df[column] = df[column].astype('category')

    # df[] = df.columns.astype('category')
    df = df.apply(lambda x: x.cat.codes)
    # print df
    return df

if __name__ == '__main__':
    main()
