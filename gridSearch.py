import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 30, 50]
}

dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
feature_names = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
X = df[feature_names].values
y = df['Survived'].values

gs.fit(X, y)

print("best params:", gs.best_params_)

print("best score:", gs.best_score_)

#############

passengers = {'survived': 6, 'non_survived': 14}


def norm(item: dict):
    return sum(item.values())


def percentage_survived(item: dict):
    return item['survived'] / norm(item)  # persent of passenger survived


def gini_impurity(item: dict):
    return 2.0 * percentage_survived(item) * (1.0 - percentage_survived(item))


A = [{'survived': 3, 'non_survived': 2}, {'survived': 3, 'non_survived': 12}]
B = [{'survived': 6, 'non_survived': 0}, {'survived': 0, 'non_survived': 14}]
C = [{'survived': 10, 'non_survived': 0}, {'survived': 6, 'non_survived': 4}]


def information_gain(A):
    return (gini_impurity(passengers) - norm(A[0]) / norm(passengers) * gini_impurity(A[0]) - norm(A[1]) / norm(
        passengers) * gini_impurity(A[1]))

print(gini_impurity(passengers))
print(gini_impurity(A[0]))
print(gini_impurity(A[1]))

print(information_gain(A))
print(information_gain(B))
print(information_gain(C))
