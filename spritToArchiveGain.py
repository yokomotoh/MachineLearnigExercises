S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

def norm(item: list):
    return len(item)


def percentage_survived(item: list):
    return sum(item) / norm(item)  # percentage1 of passenger survived


def gini_impurity(item: list):
    return 2.0 * percentage_survived(item) * (1.0 - percentage_survived(item))

def information_gain(S: list):
    return (gini_impurity(S) - norm(A) / norm(S) * gini_impurity(A) - norm(B) / norm(S) * gini_impurity(B))

#print(gini_impurity(S))
#print(gini_impurity(A))
#print(gini_impurity(B))
print(round(information_gain(S),5))