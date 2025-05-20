import cv2
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

image = cv2.imread("imageonline.jpg", cv2.IMREAD_GRAYSCALE)
M, N = image.shape 

r = 3.99 
x0 = 0.5  
num_pixels = M * N 
chaotic_sequence = logistic_map(r, x0, num_pixels)
chaotic_sequence = (chaotic_sequence * 255).astype(np.uint8)
chaotic_matrix = chaotic_sequence.reshape(M, N)

encrypted_image = cv2.bitwise_xor(image, chaotic_matrix)


cv2.imwrite("encrypted_imageonline.jpg", encrypted_image)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis("off")

plt.show()
