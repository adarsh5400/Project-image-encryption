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

def fitness_function(encrypted_image):
    hist = np.histogram(encrypted_image, bins=256)[0]
    entropy = -np.sum((hist/np.sum(hist)) * np.log2(hist + 1e-9))
    correlation = np.corrcoef(encrypted_image.flatten(), np.roll(encrypted_image.flatten(), 1))[0, 1]  # Pixel correlation
    return entropy - abs(correlation)

def pso_optimize(image, num_particles=5, iterations=10, r=3.99):
    M, N = image.shape
    num_pixels = M * N 

    particles = []
    fitness_scores = []

    for _ in range(num_particles):
        x0 = random.uniform(0, 1)
        chaotic_seq = logistic_map(r, x0, num_pixels) 
        chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N) 
        encrypted_image = cv2.bitwise_xor(image, chaotic_seq) 
        
        particles.append(encrypted_image)
        fitness_scores.append(fitness_function(encrypted_image))

    for _ in range(iterations):
        for i in range(num_particles):
            new_x0 = random.uniform(0, 1)
            chaotic_seq = logistic_map(r, new_x0, num_pixels)
            chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
            new_encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
            
            new_fitness = fitness_function(new_encrypted_image)

            if new_fitness > fitness_scores[i]:
                particles[i] = new_encrypted_image
                fitness_scores[i] = new_fitness

    best_index = np.argmax(fitness_scores)
    return particles[best_index]

image = cv2.imread("imageonline.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))

best_encrypted_image = pso_optimize(image)

cv2.imwrite("best_encrypted_image.jpg", best_encrypted_image)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_encrypted_image, cmap='gray')
plt.title("Optimized Encrypted Image")
plt.axis("off")

plt.show()
