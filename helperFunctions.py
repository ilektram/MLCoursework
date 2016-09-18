import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split


# Swiss army knife function to organize the data
def encode(df, col_list):
    for col in col_list:
        le = LabelEncoder().fit(df[col])
        classes = list(le.classes_)  # save column names for submission
        df[col] = le.transform(df[col])
    return df, classes


def preprocess(df, col_list):
    df.fillna('type 0', inplace=True)
    df, classes = encode(df, col_list)
    return df, classes


def training_testing_sets(df):
    X = df[col_list]
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


def stratified_split(df):
    sss = StratifiedShuffleSplit(df, 10, test_size=0.3, random_state=23)
    X = df[col_list]
    y = df['outcome']
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]