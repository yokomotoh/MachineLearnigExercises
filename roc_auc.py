import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
# print(df.head)
df['male'] = df['Sex']=='male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("model1 auc score: ", roc_auc_score(y_test, y_pred_proba1[:,1]))

model2 = LogisticRegression()
model2.fit(X_train[:,0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:,0:2])
print("model2 auc score: ", roc_auc_score(y_test, y_pred_proba2[:,1]))

fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_proba1[:, 1])
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba2[:, 1])

import matplotlib.pyplot as plt

plt.plot(fpr1, tpr1)
plt.plot(fpr2, tpr2)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("accuracy: {0:.5f}".format(accuracy_score(y_test, y_pred)))
print("Precision: {0:.5f}".format(precision_score(y_test, y_pred)))
print("recall: {0:.5f}".format(recall_score(y_test, y_pred)))
print("f1: {0:.5f}".format(f1_score(y_test, y_pred)))