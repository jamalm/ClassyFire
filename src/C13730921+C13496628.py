# import gaussian naive bayes model
from sklearn.naive_bayes import GaussianNB
# import splitting data
from sklearn.model_selection import train_test_split
# for preprocessing
from sklearn import preprocessing
# to get accuracy for testing
from sklearn.metrics import accuracy_score
#for data storage
import pandas as pd
import numpy
numpy.set_printoptions(threshold=numpy.nan)

def main():
    # paths to data files
    train_path = r'.\data\trainingset.txt'
    test_path = r'.\data\queries.txt'
    # read in training data and test data
    traindata = pd.read_csv(train_path, header=None)
    testdata = pd.read_csv(test_path, header=None)
    # reset the index after dropping features
    traindata = oversample(traindata)

    testlabels = testdata.loc[:, 0]

    #traindata[0] = traindata.index
    #testdata[0] = testdata.index

    # revised datasets - encoded and scaled
    traindata_rev = encode_data(traindata)
    testdata_rev = encode_data(testdata)
    # fit model and make prediction, return model and prediction results
    pred_results = predict_outcome(traindata_rev, testdata_rev)
    # format and output the solution to a file
    output_result(testlabels, pred_results)


def encode_data(df):
    revised = df

    revised = modify_data(df)
    encoder = preprocessing.LabelEncoder()
    revised.columns = range(14)

    cat_index =[2,3,4,6,7,8,10,12,13]

    for i in cat_index:
        revised[i] = encoder.fit_transform(df[i])

    for each in range(14):
        mean,std = revised[each].mean(), revised[each].std()
        revised.iloc[:, each] = (revised[each] - mean)/std
    return revised


def prepare_data(data, testdata):
    ftrain = data.values[:,:13]
    ttrain = data.values[:,13]
    ftest = testdata.values[:,:13]
    ttest = testdata.values[:,13]
    return ftrain, ftest, ttrain, ttest


def predict_outcome(data, testdata):
    features_train, features_test, target_train, target_test = prepare_data(data, testdata)
    clf = GaussianNB()
    clf.fit(features_train.astype(int), target_train.astype(int))
    target_pred = clf.predict(features_test)
    #print target_pred

    accuracy = accuracy_score(target_test.astype(int), target_pred.astype(int))
    return target_pred


def oversample(data):
    typebs = data.loc[data[17] == 'TypeB'].copy()
    result = [data, typebs, typebs, typebs]
    data = pd.concat(result)

    return data



def output_result(testlabels, pred_res):
    output = open('./solutions/C13730921+C13496628.txt', 'w')
    for i in range(0, len(testlabels)):
        if pred_res[i] == 0:
            output.write(testlabels[i] + ',"TypeA"\n')
        else:
            output.write(testlabels[i] + ',"TypeB"\n')
    output.close()

def modify_data(df):
    revised = df
    # drop previous, duration, campaign and convert pdays to binary category
    # drop default
    cols = [15,13,12,5]
    for i in cols:
        revised.drop(revised.columns[i], axis=1, inplace=True)
    #reset index
    revised[0] = revised.index

    #print revised
    simon = preprocessing.binarize(revised.loc[:, 14], 0, copy=True)
    revised[14] = simon[0, :]
    return revised



if __name__ == '__main__':
    main()
