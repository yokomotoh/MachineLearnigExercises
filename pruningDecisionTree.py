import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex']=='male'
feature_names = ['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
dt.fit(X_train, y_train)

# print(dt.predict([[3, True, 22, 1, 0, 7.25]]))
print("accuracy: ", dt.score(X_test, y_test))
y_pred = dt.predict(X_test)
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))