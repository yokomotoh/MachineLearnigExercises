# tp, fp, fn, tn = [int(x) for x in input().split()]
tp, fp, fn, tn = [233, 65, 109, 480]

accuracy = (tp + tn)/(tp+fp+fn+tn)
print(round(accuracy,4))
precision = tp / (tp + fp)
print(round(precision, 4))
recall = tp / (tp + fn)
print(round(recall, 4))
f1 = 2*(precision * recall) / (precision + recall)
print(round(f1, 4))