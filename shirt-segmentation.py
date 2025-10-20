import cv2
import numpy as np
from matplotlib import pyplot as plt

# ================================
# Step 0: Load Image
# ================================
image = cv2.imread('train_mini1/clothing_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV
hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, w, _ = hsv_img.shape

# ================================
# Step 1: Create color + position features
# ================================
x, y = np.meshgrid(np.arange(w), np.arange(h))
x = x / w * 255
y = y / h * 255

features = np.concatenate([
    hsv_img.reshape(-1, 3),
    np.stack([x, y], axis=2).reshape(-1, 2)
], axis=1).astype(np.float32)

# ================================
# Step 2: Initialize parameters
# ================================
k = 10
epsilon = 1.0
max_iter = 100

np.random.seed(42)
initial_indices = np.random.choice(len(features), k, replace=False)
means = features[initial_indices]

# ================================
# Step 3–4: Run K-Means manually
# ================================
for iteration in range(max_iter):
    distances = np.linalg.norm(features[:, np.newaxis] - means, axis=2)
    labels = np.argmin(distances, axis=1)

    new_means = np.array([
        features[labels == i].mean(axis=0) if np.any(labels == i) else means[i]
        for i in range(k)
    ])

    diff = np.linalg.norm(new_means - means)
    if diff < epsilon:
        print(f"Converged after {iteration+1} iterations.")
        break
    means = new_means

# ================================
# Step 5: Identify likely shirt cluster
# ================================
center_region = labels.reshape(h, w)[h//3:2*h//3, w//3:2*w//3]
shirt_cluster = np.bincount(center_region.flatten()).argmax()
print(f"Likely shirt cluster: {shirt_cluster}")

# ================================
# Step 6: Create mask for that cluster
# ================================
mask = (labels == shirt_cluster).reshape(h, w)

shirt_only = image.copy()
shirt_only[~mask] = 0  # black out everything else

# Save result as new image
output_filename = 'shirt_extracted.jpg'
cv2.imwrite(output_filename, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))
print(f"Saved isolated shirt image as: {output_filename}")

# ================================
# Step 7: Display results
# ================================
segmented_pixels = means[labels][:, :3].reshape(h, w, 3)
segmented_image = cv2.cvtColor(np.uint8(segmented_pixels), cv2.COLOR_HSV2RGB)

plt.figure(figsize=(15, 7))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Segmented Image (k={k})")
plt.imshow(segmented_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Likely Shirt Region")
plt.imshow(shirt_only)
plt.axis('off')

plt.show()