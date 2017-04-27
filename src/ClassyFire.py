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
import numpy as np
def main():
    # paths to data files
    train_path = r'.\data\trainingset.txt'
    test_path = r'.\data\queries.txt'
    # read in training data
    traindata = pd.read_csv(train_path, header=None)

    # drop the duration feature
    # traindata.drop(traindata.columns[12], axis=1, inplace=True)
    # reset the index
    # traindata[0] = traindata.index

    # revised trained set - encoded and scaled
    traindata_rev = encode_data(traindata)
    # fit model and make prediction, return model and the accuracy of it
    accuracy, clf = predict_outcome(traindata_rev)

    print("Accuracy of Gaussian NB: {0}").format(accuracy)

def encode_data(df):
    revised = df

    revised = modify_data(df)
    le = preprocessing.LabelEncoder()
    revised.columns = range(14)

    print revised
    cat_index =[2,3,4,6,7,8,10,12,13]

    for i in cat_index:
        revised[i] = le.fit_transform(df[i])

    scaled_features = {}
    for each in range(14):
        mean,std = revised[each].mean(), revised[each].std()
        scaled_features[each] = [mean, std]
        revised.iloc[:, each] = (revised[each] - mean)/std
    return revised

def prepare_data(data):
    features = data.values[:,:13]
    target = data.values[:,13]
    ftrain, ftest, ttrain, ttest = train_test_split(features, target, test_size=0.2, random_state= 10)
    return (ftrain, ftest, ttrain, ttest)

def predict_outcome(data):
    features_train, features_test, target_train, target_test = prepare_data(data)
    clf = GaussianNB()
    clf.fit(features_train.astype(int), target_train.astype(int))
    target_pred = clf.predict(features_test)
    np.set_printoptions(threshold='nan')
    print target_pred
    accuracy = accuracy_score(target_test.astype(int), target_pred.astype(int))
    return accuracy, clf

def modify_data(df):
    revised = df
    # drop previous, duration, campaign and convert pdays to binary category
    # drop default
    cols = [15,13,12,5]
    for i in cols:
        revised.drop(revised.columns[i], axis=1, inplace=True)
    #reset index
    revised[0] = revised.index
    return revised

if __name__ == '__main__':
    main()
