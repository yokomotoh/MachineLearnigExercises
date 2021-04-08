import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
print(df.head)

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("predict proba: ")
# print(model.predict_proba(X_test))

# y_pred = model.predict_proba(X_test)[:, 1] > 0.75
# from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, precision_recall_fscore_support, roc_curve

y_pred_proba = model.predict_proba(X_test)
# print(roc_curve(y_test, y_pred_proba[:, 1]))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])



# sensitivity_score = recall_score
# print("sensitivity: ", sensitivity_score(y_test, y_pred))


def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]


# print("specificity: ", specificity_score(y_test, y_pred))

# print(precision_recall_fscore_support(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()