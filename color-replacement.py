import cv2
import numpy as np
from matplotlib import pyplot as plt

# ================================
# SETTINGS
# ================================
input_image = 'shirt_extracted.jpg'
output_image = 'shirt_colored.jpg'

# ================================
# Step 0: Let user choose color
# ================================
color_dict = {
    'red': 0,       # hue for red
    'yellow': 30,   # hue for yellow
    'green': 60,    # hue for green
    'cyan': 90,
    'blue': 120,
    'magenta': 150
}

print("Available colors:", list(color_dict.keys()))
chosen_color = input("Enter color name: ").strip().lower()

if chosen_color not in color_dict:
    print("Color not recognized! Defaulting to green.")
    new_hue = 60
else:
    new_hue = color_dict[chosen_color]

# ================================
# Step 1: Load segmented shirt image
# ================================
shirt_only = cv2.imread(input_image)
if shirt_only is None:
    raise FileNotFoundError(f"Could not load {input_image}!")
shirt_only = cv2.cvtColor(shirt_only, cv2.COLOR_BGR2RGB)
h, w, _ = shirt_only.shape

# Create mask: any non-black pixel is shirt
mask = np.any(shirt_only != [0, 0, 0], axis=2)

# ================================
# Step 2: Convert to HSV
# ================================
hsv_shirt = cv2.cvtColor(shirt_only, cv2.COLOR_RGB2HSV)

# ================================
# Step 3: Change hue only on shirt pixels
# ================================
hsv_shirt[..., 0][mask] = new_hue

# Convert back to RGB
shirt_colored = cv2.cvtColor(hsv_shirt, cv2.COLOR_HSV2RGB)

# ================================
# Step 4: Save and display
# ================================
cv2.imwrite(output_image, cv2.cvtColor(shirt_colored, cv2.COLOR_RGB2BGR))
print(f"Saved colored shirt image as: {output_image}")

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Segmented Shirt")
plt.imshow(shirt_only)
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Shirt Colored ({chosen_color})")
plt.imshow(shirt_colored)
plt.axis('off')

plt.show()