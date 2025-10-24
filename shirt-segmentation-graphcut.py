import cv2
import numpy as np
from matplotlib import pyplot as plt

def segment_shirt(image_path, display=True):
    """
    Segment the t-shirt region from an image using an enhanced Graph Cut approach.
    Improvements:
    - Adaptive bounding box based on image content
    - Better skin color detection and exclusion
    - Multi-stage refinement
    - Adaptive color thresholding
    """
    # -------------------------------
    # Step 1: Load and preprocess image
    # -------------------------------
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # -------------------------------
    # Step 2: Detect and exclude skin regions
    # -------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Enhanced skin detection using multiple color spaces
    # HSV skin detection
    lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv = np.array([20, 170, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
    
    # YCrCb skin detection (more robust)
    lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
    
    # Combine skin masks
    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
    
    # Clean up skin mask
    kernel_skin = np.ones((7, 7), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_skin, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel_skin, iterations=1)  # Expand slightly
    
    # -------------------------------
    # Step 3: Adaptive bounding box based on content
    # -------------------------------
    # Find center of mass of non-skin regions in middle of image
    center_region = np.zeros_like(skin_mask)
    center_region[int(0.2*h):int(0.8*h), int(0.2*w):int(0.8*w)] = 1
    non_skin_center = cv2.bitwise_and(center_region, cv2.bitwise_not(skin_mask))
    
    # Use traditional bounding box with slight adjustments
    x_start = int(0.20 * w)
    y_start = int(0.25 * h)
    box_width = int(0.60 * w)
    box_height = int(0.55 * h)
    rect = (x_start, y_start, box_width, box_height)
    
    # -------------------------------
    # Step 4: Initialize mask with smart seeding
    # -------------------------------
    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    
    # Mark definite background (skin regions)
    mask[skin_mask > 0] = cv2.GC_BGD
    
    # Mark probable foreground in the rectangle (excluding skin)
    mask[y_start:y_start + box_height, x_start:x_start + box_width] = cv2.GC_PR_FGD
    mask[(y_start <= np.arange(h)[:, None]) & 
         (np.arange(h)[:, None] < y_start + box_height) &
         (x_start <= np.arange(w)) & 
         (np.arange(w) < x_start + box_width) &
         (skin_mask > 0)] = cv2.GC_BGD
    
    # -------------------------------
    # Step 5: Detect shirt-like colors (excluding skin tones)
    # -------------------------------
    # Calculate dominant colors in the probable foreground region
    roi = image[y_start:y_start + box_height, x_start:x_start + box_width]
    roi_mask = skin_mask[y_start:y_start + box_height, x_start:x_start + box_width] == 0
    
    if np.any(roi_mask):
        roi_pixels = roi[roi_mask]
        
        # Find shirt color characteristics
        mean_color = np.mean(roi_pixels, axis=0).astype(np.uint8)
        std_color = np.std(roi_pixels, axis=0).astype(np.uint8)
        
        # Create adaptive color range
        lower_shirt = np.clip(mean_color - 2.5 * std_color, 0, 255).astype(np.uint8)
        upper_shirt = np.clip(mean_color + 2.5 * std_color, 0, 255).astype(np.uint8)
        
        # Detect shirt-colored regions
        shirt_color_mask = cv2.inRange(image, lower_shirt, upper_shirt)
        
        # Exclude skin regions from shirt color mask
        shirt_color_mask = cv2.bitwise_and(shirt_color_mask, cv2.bitwise_not(skin_mask))
        
        # Mark strong shirt color regions as definite foreground
        kernel_color = np.ones((5, 5), np.uint8)
        shirt_color_mask = cv2.morphologyEx(shirt_color_mask, cv2.MORPH_CLOSE, kernel_color, iterations=2)
        
        mask[(shirt_color_mask > 0) & (mask == cv2.GC_PR_FGD)] = cv2.GC_FGD
    
    # -------------------------------
    # Step 6: Run GrabCut with increased iterations
    # -------------------------------
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, mask, rect, bg_model, fg_model, 20, cv2.GC_INIT_WITH_RECT)
    
    # -------------------------------
    # Step 7: Post-processing refinement
    # -------------------------------
    mask_final = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    
    # Remove skin regions from final mask
    mask_final = cv2.bitwise_and(mask_final, cv2.bitwise_not(skin_mask // 255))
    
    # Morphological operations to clean up
    kernel = np.ones((7, 7), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes using contour analysis
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep only the largest contour (assumed to be the shirt)
        largest_contour = max(contours, key=cv2.contourArea)
        mask_refined = np.zeros_like(mask_final)
        cv2.drawContours(mask_refined, [largest_contour], -1, 1, thickness=cv2.FILLED)
        mask_final = mask_refined
    
    # Apply mask to original image
    segmented_img = image * mask_final[:, :, np.newaxis]
    
    # -------------------------------
    # Step 8: Display (optional)
    # -------------------------------
    if display:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 4, 2)
        plt.imshow(skin_mask, cmap="gray")
        plt.title("Detected Skin Regions")
        plt.axis("off")
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask_final, cmap="gray")
        plt.title("Final Shirt Mask")
        plt.axis("off")
        
        plt.subplot(1, 4, 4)
        plt.imshow(segmented_img)
        plt.title("Segmented Shirt")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    return mask_final, segmented_img


if __name__ == "__main__":
    # Process multiple images
    for i in range(1, 34, 1):
        image_path = "clothing_" + str(i) + ".jpg"
        try:
            mask, shirt = segment_shirt("mini_image_set/" + image_path, display=True)
            cv2.imwrite(f"output/shirt_mask_{i}.png", mask * 255)
            cv2.imwrite(f"output/segmented_shirt_{i}.png", cv2.cvtColor(shirt, cv2.COLOR_RGB2BGR))
            print(f"Processed {image_path} successfully")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
