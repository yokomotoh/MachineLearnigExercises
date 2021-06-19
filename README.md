# MachineLearnigExercises

Machine Learning exercises on PyCharm

- machineLearnig.py

## Classification

prediction probability

precision and recall

plot roc curve

- roc_auc.py

roc_curve

roc auc score

scores: accurary, precision, recall, f1


- k-foldCrossValidation.py



- bobTheBuilder.py

Building a Logistic Regression model.

Task
You are given a feature matrix and a single datapoint to predict. Your job will be to build a Logistic Regression model with the feature matrix and make a prediction (1 or 0) of the single datapoint.

Input Format
First line: Number of data points in the feature matrix (n)
Next n lines: Values of the row in the feature matrix, separated by spaces
Next line: Target values separated by spaces
Final line: Values (separated by spaces) of a single datapoint without a target value

Output Format
Either 1 or 0

## Model Evaluation

- modelComparison.py

compare three models

- matrixWelcome.py

Calculating Evaluation Metrics using the Confusion Matrix.

Task
You will be given the values of the confusion matrix (true positives, false positives, false negatives, and true negatives). Your job is to compute the accuracy, precision, recall and f1 score and print the values rounded to 4 decimal places. To round, you can use round(x, 4).

Input Format
The values of tp, fp, fn, tn, in that order separated by spaces

Output Format
Each value on its own line, rounded to 4 decimal places, in this order:
accuracy, precision, recall, f1 score

## Decision Tree Model

- decisionTree.py

- decisionTreesComparison.py

- pruningDecisionTree.py

- spritToArchiveGain.py

Calculate Information Gain.

Task
Given a dataset and a split of the dataset, calculate the information gain using the gini impurity.

The first line of the input is a list of the target values in the initial dataset. The second line is the target values of the left split and the third line is the target values of the right split.

Round your result to 5 decimal places. You can use round(x, 5).

Input Format
Three lines of 1's and 0's separated by spaces

Output Format
Float (rounded to 5 decimal places)

## Random Forest Model

- tuningRandomForest.py

- featureImportances.py

- aForestOfTrees.py

Build a Random Forest model.

Task
You will be given a feature matrix X and target array y. Your task is to split the data into training and test sets, build a Random Forest model with the training set, and make predictions for the test set. Give the random forest 5 trees.

You will be given an integer to be used as the random state. Make sure to use it in both the train test split and the Random Forest model.

Input Format
First line: integer (random state to use)
Second line: integer (number of datapoints)
Next n lines: Values of the row in the feature matrix, separated by spaces
Last line: Target values separated by spaces

Output Format
Numpy array of 1's and 0's

## NewralNetworks

- neuralNetArtificialDataset.py

- neuralNetOpenML.py

- neuralNetMLPClassifier.py

- neuralNetPredictHandwrittenDigits.py

- neuralNetSigmoidFunction.py
