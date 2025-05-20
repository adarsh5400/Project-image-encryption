import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

def encrypt_image(image, x0, r=3.99):
    M, N = image.shape
    num_pixels = M * N
    chaotic_seq = logistic_map(r, x0, num_pixels)
    chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)

    encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
    
    return encrypted_image, chaotic_seq

def decrypt_image(encrypted_image, chaotic_seq):
    decrypted_image = cv2.bitwise_xor(encrypted_image, chaotic_seq)
    return decrypted_image

image = cv2.imread("imageonline.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

x0 = 0.56789
r = 3.99 

encrypted_image, chaotic_seq = encrypt_image(image, x0, r)

cv2.imwrite("encrypted_image.jpg", encrypted_image)

decrypted_image = decrypt_image(encrypted_image, chaotic_seq)

cv2.imwrite("decrypted_image.jpg", decrypted_image)

plt.figure(figsize=(10,7))

plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(decrypted_image, cmap='gray')
plt.title("Decrypted Image")
plt.axis("off")

plt.show()
