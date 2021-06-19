from math import exp

w1, w2, b, x1, x2 = [float(x) for x in input().split()]

# print(w1, w2, b, x1, x2)
y = 1 / (1 + exp(-1*(w1*x1 + w2*x2 + b)))
print(round(y, 4))