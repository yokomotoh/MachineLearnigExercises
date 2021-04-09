from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

kf = KFold(n_splits=5, shuffle=True)
X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

def model_score(X, y, kf):
    score = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        score.append(model.score(X_test, y_test))

        y_pred = model.predict(X_test)
        # print("accuracy: {0:.5f}".format(accuracy_score(y_test, y_pred)))
        # print("Precision: {0:.5f}".format(precision_score(y_test, y_pred)))
        # print("recall: {0:.5f}".format(recall_score(y_test, y_pred)))
        # print("f1: {0:.5f}".format(f1_score(y_test, y_pred)))
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append((f1_score(y_test, y_pred)))
    #
    # print(score)
    print([np.mean(score), np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)])

print("model ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']")
model_score(X1, y, kf)
print("model ['Pclass', 'male', 'Age']")
model_score(X2, y, kf)
print("model ['Fare', 'Age']")
model_score(X3, y, kf)

# test example
model = LogisticRegression()
model.fit(X1, y)
print(model.predict([[3, False, 25, 0, 1, 2]]))
