import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# ================================
# Step 0: Load and downsample image
# ================================
input_path = 'train_mini1/clothing_2.jpg'
output_path = 'shirt_extracted_meanshift_fast.jpg'

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Could not read {input_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
scale = 0.25  # downsample for speed
small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
h_s, w_s, _ = small_img.shape

# Convert to HSV (better for color clustering)
hsv_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)
features = hsv_img.reshape(-1, 3).astype(np.float32)

# ================================
# Step 1: Mean Shift Parameters
# ================================
window_radius = 60.0
epsilon = 1.0
max_iter = 8
spatial_window = 25

means = features.copy()

# ================================
# Step 2: Mean Shift Iteration
# ================================
for iteration in range(max_iter):
    print(f"\nIteration {iteration + 1}/{max_iter}")
    new_means = np.zeros_like(means)
    max_shift = 0

    for i in tqdm(range(len(means)), desc="Pixels processed", ncols=80):
        m = means[i]
        y_idx = i // w_s
        x_idx = i % w_s

        y_min = max(0, y_idx - spatial_window)
        y_max = min(h_s, y_idx + spatial_window + 1)
        x_min = max(0, x_idx - spatial_window)
        x_max = min(w_s, x_idx + spatial_window + 1)

        local_pixels = hsv_img[y_min:y_max, x_min:x_max].reshape(-1, 3).astype(np.float32)
        dist = np.linalg.norm(local_pixels - m, axis=1)
        in_window = local_pixels[dist < window_radius]

        if len(in_window) > 0:
            mean_val = np.mean(in_window, axis=0)
            new_means[i] = mean_val
            shift = np.linalg.norm(mean_val - m)
            max_shift = max(max_shift, shift)
        else:
            new_means[i] = m

    print(f"Max shift this iteration: {max_shift:.2f}")
    means = new_means
    if max_shift < epsilon:
        print("Converged early.")
        break

# ================================
# Step 3: Cluster Merging
# ================================
cluster_thresh = 45.0
unique_means = []
labels = np.zeros(len(means), dtype=int)

for i in range(len(means)):
    m = means[i]
    found = False
    for j, um in enumerate(unique_means):
        if np.linalg.norm(m - um) < cluster_thresh:
            labels[i] = j
            found = True
            break
    if not found:
        unique_means.append(m)
        labels[i] = len(unique_means) - 1

unique_means = np.array(unique_means)

# ================================
# Step 4: Find likely shirt cluster
# ================================
labels_img = labels.reshape(h_s, w_s)
center_region = labels_img[h_s // 3:2 * h_s // 3, w_s // 3:2 * w_s // 3]
shirt_cluster = np.bincount(center_region.flatten()).argmax()
print(f"Likely shirt cluster: {shirt_cluster}")

# ================================
# Step 5: Create mask and upsample
# ================================
mask_small = (labels_img == shirt_cluster).astype(np.uint8)

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)

# Upsample to original size
mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

# Keep only shirt region
shirt_only = image.copy()
shirt_only[mask == 0] = 0

# ================================
# Step 6: Save and Display
# ================================
cv2.imwrite(output_path, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))
print(f"\nSaved shirt-isolated image as: {output_path}")

# ================================
# Step 7: Plot segmentation
# ================================
segmentation_vis = (labels_img * (255 // (len(unique_means) + 1))).astype(np.uint8)
segmentation_vis = cv2.applyColorMap(segmentation_vis, cv2.COLORMAP_JET)

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmentation Map")
plt.imshow(segmentation_vis)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Isolated Shirt")
plt.imshow(shirt_only)
plt.axis('off')

plt.tight_layout()
plt.show()
