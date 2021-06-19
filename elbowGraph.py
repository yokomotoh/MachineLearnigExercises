from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

n_estimaotors = list(range(1,101))
param_grid = {
    'n_estimators': n_estimaotors
}

cancer_data = load_breast_cancer()
# print(cancer_data['DESCR'])

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
# print(df)
X = df[cancer_data.feature_names].values
y = df['target'].values

"""
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)

gs.fit(X,y)
scores = gs.cv_results_['mean_test_score']
print(scores)


import matplotlib.pyplot as plt
plt.plot(n_estimaotors, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()
"""

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)

