import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.model_selection import KFold


df_cancer = load_breast_cancer()
cols = list(df_cancer.feature_names)
cols.append('target')
print(df_cancer.data)

train_df = pd.DataFrame(np.c_[df_cancer.data, df_cancer.target], columns=cols)

print(f" Columns in dataset: {cols}")
print(train_df.head())

y = train_df['target']
X = train_df.drop('target', axis=1)
print(X.shape)

folds= KFold(n_splits=5,shuffle=True,random_state=12)
for fold,(train_idx,test_idx) in enumerate(folds.split(X,y)):
    print("Fold", fold)
    x_train=X.iloc[train_idx]
    y_train=y[train_idx]
    x_val=X.iloc[test_idx]
    y_val=y[test_idx]

    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    print(f'Accuracy score: {accuracy_score(y_val, pred)}')
    print(f"Confusion matrix: \n{confusion_matrix(y_val, pred)}")
