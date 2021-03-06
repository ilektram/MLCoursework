import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn import linear_model, decomposition, datasets
from sklearn.grid_search import GridSearchCV

from sklearn.externals import joblib


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


def training_testing_sets(df, col_list):
    X = df[col_list]
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def stratified_split(df, col_list):
    sss = StratifiedShuffleSplit(df, 10, test_size=0.3, random_state=23)
    X = df[col_list]
    y = df['outcome']
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def logistic_model_search(X_train, y_train):
    logistic = linear_model.LogisticRegression(verbose=True, class_weight='balanced', random_state=0, penalty='l2')
    # Define parameters to search through
    logistic_param_grid = {
                            'tol': np.arange(0, 5, .5),
                            'C': [.1, .5, 1, 1.1, 1.5, 5, 10],
                            'max_iter': np.arange(100, 500, 1000),
                            'solver': ['newton-cg', 'lbfgs', 'sag']
    }
    # Split training data to training/validation sets and test hyperparameters
    logistic_gridSearch = GridSearchCV(logistic,
                                       logistic_param_grid,
                                       scoring=None,
                                       fit_params=None,
                                       cv=5,
                                       verbose=10,
                                       error_score='raise',
                                       n_jobs=3,
                                       pre_dispatch=6)
    # Run hyperparameter optimisation
    logistic_gridSearch.fit(X_train, y_train)
    # Print test results
    print("Logistic Regression Parameter Fitting: ", logistic_gridSearch.grid_scores_)
    print("Logistic Regression Best Estimator: ", logistic_gridSearch.best_estimator_)
    print("Logistic Regression Best Score: ", logistic_gridSearch.best_score_)
    print("Logistic Regression Best Parameters: ", logistic_gridSearch.best_params_)
    print("Logistic Regression Scorer: ", logistic_gridSearch.scorer_)

    # Save as pickle
    joblib.dump(logistic_gridSearch, 'logisticModel.pkl')
    return logistic_gridSearch

