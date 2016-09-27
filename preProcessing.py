
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dateutil.parser import parse
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from helperFunctions import preprocess, stratified_split, training_testing_sets, logistic_model_search


act_train_df = pd.read_csv('act_train.csv')
#print(act_train_df.columns)
people_df = pd.read_csv('people.csv')
#print(people_df.columns)
merged_df = pd.merge(act_train_df, people_df, on='people_id')
#print(merged_df.columns)

col_list = ['activity_category', 'char_1_x',
       'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x',
       'char_8_x', 'char_9_x', 'char_10_x', 'char_1_y', 'group_1',
       'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y',
       'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
       'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18',
       'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24',
       'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30',
       'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
       'char_37', 'char_38']

merged_df, classes = preprocess(merged_df, col_list)

X_train, X_test, y_train, y_test = training_testing_sets(merged_df, col_list)
###############################################################################


logistic_Model = logistic_model_search(X_train, y_train)