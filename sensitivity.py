import numpy as np

def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence

r = 3.99 
x0 = 0.5000000 
x1 = 0.5000001
n = 10 

seq1 = logistic_map(r, x0, n)
seq2 = logistic_map(r, x1, n)

print("Sequence with x0 =", seq1)
print("Sequence with x1 =", seq2)
