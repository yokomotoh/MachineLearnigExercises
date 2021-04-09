
n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]


from sklearn.linear_model import LogisticRegression
"""
X = [[1.0, 3.0],[3.0, 5.0],[5.0, 7.0],[3.0, 1.0],[5.0, 3.0],[7.0, 5.0]]
y = [1, 1, 1, 0, 0, 0]
datapoint = [2, 4]
"""
print(X)
print(y)
print(datapoint)

"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

print("X_train: ", X_train)
print("y_train: ", y_train)
print("X_test: ", X_test)
print("y_test: ", y_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("predict: ", y_pred)

print("predict proba: ")
print(model.predict_proba(X_test))

# y_pred = model.predict_proba(X_test)[:, 1] > 0.75
"""

model = LogisticRegression()
model.fit(X, y)
datapoint_pred = model.predict([datapoint])
print(datapoint_pred[0])

import matplotlib.pyplot as plt

plt.scatter([x[0] for x in X ], [x[1] for x in X ], c=y)
plt.scatter(datapoint[0], datapoint[1])
plt.plot([0, 8], [0, 8], linestyle='--')
plt.xlim([0.0, 8.0])
plt.ylim([0.0, 8.0])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()