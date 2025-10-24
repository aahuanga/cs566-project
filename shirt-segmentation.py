# from sklearn.cluster import KMeans
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# def process_image(image_path):
#     # Load and preprocess image
#     path = 'train_mini1/' + image_path
#     image = cv2.imread(path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     h, w, _ = hsv_img.shape

#     # Create meshgrid for position features
#     x, y = np.meshgrid(np.arange(w), np.arange(h))
#     x = x / w
#     y = y / h

#     # Convert to LAB color space
#     lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

#     # Combine features
#     hsv_weight = 1.0
#     pos_weight = 1.0
#     features = np.concatenate([
#         lab_img.reshape(-1, 3) * hsv_weight,
#         np.stack([x, y], axis=2).reshape(-1, 2) * pos_weight
#     ], axis=1)

#     # Apply KMeans
#     k = 6
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
#     kmeans.fit(features)
#     labels = kmeans.labels_

#     # Identify likely shirt cluster
#     center_region = labels.reshape(h, w)[h//3:2*h//3, w//3:2*w//3]
#     shirt_cluster = np.bincount(center_region.flatten()).argmax()

#     # Create mask for shirt cluster
#     mask = (labels == shirt_cluster).reshape(h, w)
#     shirt_only = image.copy()
#     shirt_only[~mask] = 0

#     # Save and display results
#     output_filename = 'shirt_extracted.jpg'
#     cv2.imwrite(output_filename, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))

#     segmented_pixels = kmeans.cluster_centers_[labels][:, :3].reshape(h, w, 3)
#     segmented_image = cv2.cvtColor(np.uint8(segmented_pixels), cv2.COLOR_HSV2RGB)

#     plt.figure(figsize=(15, 7))
#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(image)
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title(f"Segmented Image (k={k})")
#     plt.imshow(segmented_image)
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title("Likely Shirt Region")
#     plt.imshow(shirt_only)
#     plt.axis('off')

#     plt.show()

# for i in range(1, 34, 1):
#     image_path = "clothing_" + str(i) + ".jpg"
#     process_image(image_path)

from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt


def process_image(image_path):
    # Load and preprocess image
    path = 'mini_image_set/' + image_path
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Create meshgrid for position features
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x / w
    y = y / h

    # Convert to LAB color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Combine features
    hsv_weight = 1.0
    pos_weight = 3.0
    features = np.concatenate([
        lab_img.reshape(-1, 3) * hsv_weight,
        np.stack([x, y], axis=2).reshape(-1, 2) * pos_weight
    ], axis=1)

    # Apply KMeans
    k = 6
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_

    # Identify likely shirt cluster
    center_region = labels.reshape(h, w)[h//3:2*h//3, w//3:2*w//3]
    shirt_cluster = np.bincount(center_region.flatten()).argmax()

    # Create mask for shirt cluster
    mask = (labels == shirt_cluster).reshape(h, w)

    # ------------------------------
    # Keep only the largest connected component
    # ------------------------------
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels > 1:
        # Ignore the background (label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels_im == largest_label)

    # Apply mask to image
    shirt_only = image.copy()
    shirt_only[~mask] = 0

    # Save and display results
    output_filename = 'shirt_extracted.jpg'
    cv2.imwrite(output_filename, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))

    # Optional: display K-means segmented image
    segmented_pixels = kmeans.cluster_centers_[labels][:, :3].reshape(h, w, 3)
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


for i in range(1, 34, 1):
    image_path = "clothing_" + str(i) + ".jpg"
    process_image(image_path)
