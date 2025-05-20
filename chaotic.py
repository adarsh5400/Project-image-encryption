import numpy as np
import matplotlib.pyplot as plt
def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence

r = 3.9 
x0 = 0.5 
n = 100 
chaotic_sequence = logistic_map(r, x0, n)
plt.figure(figsize=(10, 5))
plt.plot(chaotic_sequence, marker="o", linestyle="-", color="b")
plt.title("Chaotic Logistic Map Sequence")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.grid()
plt.show()
print("First 10 values of the chaotic sequence:", chaotic_sequence[:10])
