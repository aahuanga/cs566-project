import cv2
import numpy as np

def overlay_recolored_region(original_path, recolored_path, output_path="final_overlay.jpg", alpha=1.0): # alpha = blending strength (how much recolored vs original)

    orig = cv2.imread(original_path)
    recolored = cv2.imread(recolored_path)

    # check that both images exist
    if orig is None or recolored is None:
        raise ValueError("Error: Could not read one of the input images.")
    
    # both images need to be the same size
    if orig.shape != recolored.shape:
        raise ValueError("Original and recolored images must have the same dimensions.")

    gray = cv2.cvtColor(recolored, cv2.COLOR_BGR2GRAY) # convert recolored image to grayscale
    mask = gray > 10   # threshold to detect where pixels exist (nonblack is clothing)
    mask = mask.astype(np.uint8) # convert boolean mask to 0/1

    # blur mask to soften the edges - avoid sharp cut out lines
    mask = cv2.GaussianBlur(mask.astype(float), (5,5), 0)
    mask = np.clip(mask, 0, 1) # make sure that the mask stays between 0 and 1 range

    # create float copies of the images
    blended = orig.copy().astype(float)
    recolored_f = recolored.astype(float)

    # if mask = 0, use og pixell; if mask = 1, use recolored pixel; if mask = 0.5 (soft edge), use 50/50 of both
    blended = blended * (1 - mask[..., None]) + (alpha * recolored_f + (1 - alpha) * orig) * mask[..., None]
    blended = np.clip(blended, 0, 255).astype(np.uint8) # normalize values (valid range is 0-255)

    # save overlay result
    cv2.imwrite(output_path, blended)
    print(f"Saved final overlay to {output_path}")

    return blended

overlay_recolored_region(
    # original_path="train_mini1/clothing_1.jpg",
    original_path="mini_image_set/00007_00.jpg",
    recolored_path="00007_00_color_replaced.jpg",
    output_path="final_overlay_1.jpg",
    alpha=0.8
)
