from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True) # when only two digits, 0 and 1, n_class=2
# we will initially only be working with two digits (0 and 1),
# so we use the n_class parameter to limit the number of target values to 2.

print(X.shape, y.shape)

# we have 300 datapoints and each datapoint has 64 features.
# We have 64 features because the image is 8 x 8 pixels and we have 1 feature per pixel.
# The value is on a grayscale where 0 is black and 16 is white.

# print(X[0])
# print(y[0])
# print(X[0].reshape(8, 8))

#plt.matshow(X[3].reshape(8, 8), cmap=plt.cm.gray)
#plt.xticks(()) # remove x tick marks
#plt.yticks(()) # remove y tick marks
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

x = X_test[10]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.show()
print(mlp.predict([x]))
print(mlp.score(X_test, y_test))

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.show()
print("true value: ", incorrect_true[j])
print("predicted value: ", incorrect_pred[j])

print(incorrect.shape)
print(incorrect)
print(incorrect_true)
print(incorrect_pred)