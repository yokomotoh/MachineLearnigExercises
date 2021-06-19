from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

cancer_data = load_breast_cancer()
# print(cancer_data['DESCR'])

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
# print(df)
X = df[cancer_data.feature_names].values
y = df['target'].values

rf = RandomForestClassifier(random_state=123)

param_grid = {
    'n_estimators': [10, 25, 50, 75, 100],
    'max_features': [5, 10, 15],
    'criterion': ["gini", "entropy"]
}

gs = GridSearchCV(rf, param_grid, scoring='f1', cv=5)
gs.fit(X, y)
print("best param: ", gs.best_params_)