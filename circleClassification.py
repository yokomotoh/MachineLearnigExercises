import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

plt.scatter([itm[0] for itm in X], [itm[1] for itm in X], c=y)
#plt.xlim([-1.0, 1.0])
#plt.ylim([-1.0, 1.0])
plt.xlabel('X0')
plt.ylabel('X1')
# plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=1)

lr_scores = []
rf_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X_train, y_train)
    lr_scores.append(lr.score(X_test, y_test))
    #
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_scores.append(rf.score(X_test, y_test))

print("LR accuracy: ", np.mean(lr_scores))
print("RF accuracy: ", np.mean(rf_scores))