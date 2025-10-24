# ==========================================
# shirt_segmentation_meanshift.py
# ==========================================
import numpy as np
import cv2
from matplotlib import pyplot as plt


def process_image(image_path):
    """
    Segments the likely t-shirt region from a stock image using Mean Shift segmentation.

    Parameters
    ----------
    image_path : str
        The filename (without folder prefix) of the image to process.
    """
    # ------------------------------
    # Step 1: Load and preprocess image
    # ------------------------------
    path = 'train_mini1/' + image_path
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Convert to LAB color space (better for segmentation)
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # ------------------------------
    # Step 2: Apply Mean Shift Filtering
    # ------------------------------
    # Spatial window radius (sp) and color window radius (sr) are tunable
    # Larger sp = more spatial smoothing (fewer regions)
    # Larger sr = more color smoothing (merges similar colors)
    sp = 12
    sr = 20
    mean_shift_img = cv2.pyrMeanShiftFiltering(lab_img, sp, sr)

    # Convert back to RGB for visualization
    mean_shift_rgb = cv2.cvtColor(mean_shift_img, cv2.COLOR_LAB2RGB)

    # ------------------------------
    # Step 3: Cluster the mean-shift result into distinct labels
    # ------------------------------
    # To identify the shirt region, we’ll cluster the filtered pixels by color
    reshaped = mean_shift_img.reshape((-1, 3))
    reshaped = np.float32(reshaped)

    # Use OpenCV k-means just for post-clustering (small k since Mean Shift pre-smoothed)
    k = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    # ------------------------------
    # Step 4: Identify likely shirt cluster (largest region near image center)
    # ------------------------------
    label_img = labels.reshape(h, w)
    center_crop = label_img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    shirt_cluster = np.bincount(center_crop.flatten()).argmax()

    # Create initial mask for shirt cluster
    mask = (label_img == shirt_cluster)

    # ------------------------------
    # Step 5: Keep only the largest connected component
    # ------------------------------
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels_im == largest_label)

    # ------------------------------
    # Step 6: Apply mask to image
    # ------------------------------
    shirt_only = image.copy()
    shirt_only[~mask] = 0

    # Save results
    output_filename = 'shirt_extracted_meanshift.jpg'
    cv2.imwrite(output_filename, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))

    # ------------------------------
    # Step 7: Visualization
    # ------------------------------
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mean Shift Filtered Image")
    plt.imshow(mean_shift_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Likely Shirt Region")
    plt.imshow(shirt_only)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------
# Process all images
# --------------------------------------------
if __name__ == "__main__":
    for i in range(1, 34):
        image_path = f"clothing_{i}.jpg"
        process_image(image_path)

    # img_path = "clothing_2.jpg"  # Change to your file
    # mask, shirt = process_image(img_path)
    # cv2.imwrite("output/shirt_mask.png", mask * 255)
    # cv2.imwrite("output/segmented_shirt.png", cv2.cvtColor(shirt, cv2.COLOR_RGB2BGR))
