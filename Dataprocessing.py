
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

logistic = linear_model.LogisticRegression()

# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

act_train_df = pd.read_csv('act_train.csv')
print(act_train_df.columns)
people_df = pd.read_csv('people.csv')
print(people_df.columns)
merged_df = pd.merge(act_train_df, people_df, on='people_id', left_index='act', right_index='pe')
print(merged_df.columns)
###############################################################################
