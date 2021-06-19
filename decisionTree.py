from numpy import log

'''
p = 40 / (40 + 40)
gini = 2 * p * (1 - p)
entropy = -(p * log(p) + (1 - p) * log(1 - p))
print("gini: , ", gini)
print("entropy: ", entropy)
'''

import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head)

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
model = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
model.fit(X_train, y_train)

# print(model.predict([[3, True, 22, 1, 0, 7.25]]))
print("accuracy: ", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))