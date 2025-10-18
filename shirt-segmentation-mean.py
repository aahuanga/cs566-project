import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree

# ================================
# Step 0: Load and downsample image
# ================================
input_path = 'train_mini1/clothing_2.jpg'
output_path = 'shirt_extracted_meanshift.jpg'

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Could not read {input_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
scale = 0.4  # slightly smaller for speed
small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
h_s, w_s, _ = small_img.shape

# Convert to HSV
hsv_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)

# ================================
# Step 1: Feature Construction (HSV + Spatial)
# ================================
features = []
for y in range(h_s):
    for x in range(w_s):
        hsv = hsv_img[y, x].astype(np.float32)
        features.append(np.concatenate([hsv, [y * 0.5, x * 0.5]]))  # spatial weight
features = np.array(features)

# ================================
# Step 2: Mean Shift Parameters
# ================================
window_radius = 60.0
epsilon = 1.0
max_iter = 5

means = features.copy()
tree = cKDTree(features)

# ================================
# Step 3: Mean Shift Iteration (KD-Tree)
# ================================
for iteration in range(max_iter):
    print(f"\nIteration {iteration + 1}/{max_iter}")
    new_means = np.zeros_like(means)
    max_shift = 0

    for i in tqdm(range(len(means)), desc="Pixels processed", ncols=80):
        m = means[i]
        neighbors = tree.query_ball_point(m, r=window_radius)
        if neighbors:
            in_window = means[neighbors]
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
# Step 4: Cluster Merging
# ================================
cluster_thresh = 30.0
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
# Step 5: Find likely shirt cluster (center region)
# ================================
labels_img = labels.reshape(h_s, w_s)
center_region = labels_img[h_s // 3:2 * h_s // 3, w_s // 3:2 * w_s // 3]
shirt_cluster = np.bincount(center_region.flatten()).argmax()
print(f"Likely shirt cluster: {shirt_cluster}")

# ================================
# Step 6: Create mask and cleanup
# ================================
mask_small = (labels_img == shirt_cluster).astype(np.uint8)

# Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)

# Keep largest connected component
num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_small)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
mask_small = (labels_cc == largest_label).astype(np.uint8)

# Upsample to original size
mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

# Apply mask
shirt_only = image.copy()
shirt_only[mask == 0] = 0

# ================================
# Step 7: Save and Display
# ================================
cv2.imwrite(output_path, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))
print(f"\n Saved shirt-isolated image as: {output_path}")

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Isolated Shirt")
plt.imshow(shirt_only)
plt.axis('off')

plt.show()