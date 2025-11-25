# Segment image file using K means, picks the cluster that likely contains the shirt (by sampling the center region), keeps the largest connected 
# component of that cluster, then refines that region using GrabCut.

from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt


def process_image(image_path):
    # load image and get dimensions
    path = 'mini_image_set/' + image_path
    # path = 'train_mini1/' + image_path
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # normalized coordinate map
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x / w, y / h
    # convert RGB image to LAB color space which separates luminance (brightness) and color channels 
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # feature vector for each pixel that contains the color (LAB) and position (x, y) 
    # we can try to mess around with these values 
    hsv_weight = 1.0
    pos_weight = 3.0 # position is made 3x as important as color
    features = np.concatenate([
        lab_img.reshape(-1, 3) * hsv_weight,
        np.stack([x, y], axis=2).reshape(-1, 2) * pos_weight
    ], axis=1)

    # kmeans clustering - 6 clusters using color and position
    k = 6
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42, n_init='auto')
    kmeans.fit(features)
    labels = kmeans.labels_

    # find the cluster that is likely to be the shirt 
    center_region = labels.reshape(h, w)[h//3:2*h//3, w//3:2*w//3] # reshape labels to h, w and samples central rectangle (mmiddle third of image)
    shirt_cluster = np.bincount(center_region.flatten()).argmax() # pick most common cluster in that center patch as shirt cluster
    mask = (labels == shirt_cluster).reshape(h, w) # boolean mask where pixels in the shirt cluster are true 

    mask_uint8 = mask.astype(np.uint8) # convert boolean mask to 0/1
    mask_uint8[mask_uint8 > 0] = 1 
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, 8, cv2.CV_32S)
    # find the largest component area, excluding the background (label 0)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels_im == largest_label)

    # Grabcut refining
    # the mask is initialized to one of these: GC_BGD (0): Background, GC_FGD (1): Foreground, GC_PR_BGD (2): Probable Background, GC_PR_FGD (3): Probable Foreground
    gc_mask = np.zeros(image.shape[:2], np.uint8) 
    gc_mask[mask] = cv2.GC_PR_FGD # area identified by K-means/Connected Component as "Probable Foreground" (3)
    gc_mask[~mask] = cv2.GC_PR_BGD # area outside the K-means mask as "Probable Background" (2)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64) # updated by grabcut later
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK) # run grabcut 5 times starting from provided gcmask

    # grabcut mask ccontains refined labels where pixels labeled GC_FGD or GC_PR_FGD are treated as foreground
    final_mask_bool = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), True, False)
    shirt_only = image.copy()
    shirt_only[~final_mask_bool] = 0 # original image with non-shirt pixels zeroed out (black background)
    
    # Save and display results
    output_filename = 'shirt_extracted.jpg'
    cv2.imwrite(output_filename, cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))

    segmented_pixels = kmeans.cluster_centers_[labels][:, :3].reshape(h, w, 3)
    segmented_image = cv2.cvtColor(np.uint8(segmented_pixels), cv2.COLOR_LAB2RGB)
    
    k_means_shirt_only = image.copy()
    k_means_shirt_only[~mask] = 0

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"K-means Mask (k={k})")
    plt.imshow(k_means_shirt_only)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("GrabCut Refined Shirt Region")
    plt.imshow(shirt_only)
    plt.axis('off')

    plt.show()

    cv2.imwrite("testing.jpg", cv2.cvtColor(shirt_only, cv2.COLOR_RGB2BGR))

# base = "000"
# end = "_00.jpg"
# for i in range(40, 54, 1):
# # for i in range(40, 45, 1):
#     middle = "0" + str(i) if i < 10 else str(i)
#     image_path = base + middle + end
#     print(image_path)
#     process_image(image_path)
#     # output_filename = 'testing.jpg'
#     # plt.savefig("testing.png")


# for i in range(10):
#     image_path = "clothing_" + str(i) + ".jpg"
#     process_image(image_path)
    
image_path = "00007_00.jpg"
process_image(image_path)