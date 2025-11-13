from sklearn.cluster import KMeans
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# ================================
# IMAGE SEGMENTATION FUNCTION
# ================================
def process_image(image_path):
    path = 'mini_image_set/' + image_path
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x / w
    y = y / h

    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    hsv_weight = 1.0
    pos_weight = 3.0
    features = np.concatenate([
        lab_img.reshape(-1, 3) * hsv_weight,
        np.stack([x, y], axis=2).reshape(-1, 2) * pos_weight
    ], axis=1)

    k = 6
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=100, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_

    center_region = labels.reshape(h, w)[h//3:2*h//3, w//3:2*w//3]
    shirt_cluster = np.bincount(center_region.flatten()).argmax()

    mask = (labels == shirt_cluster).reshape(h, w)
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels_im == largest_label)

    return image, mask

# ================================
# LOAD IMAGE AND MASK
# ================================
image_path = "00047_00.jpg"
shirt_img, mask = process_image(image_path)
hsv_base = cv2.cvtColor(shirt_img, cv2.COLOR_RGB2HSV)

# ================================
# GUI FUNCTIONS
# ================================
def recolor_shirt(hue_value):
    hsv = hsv_base.copy()
    h, s, v = cv2.split(hsv)
    h[mask] = hue_value

    avg_v = np.mean(v[mask])
    avg_s = np.mean(s[mask])

    s = s.astype(np.float32)
    v = v.astype(np.float32)

    # Adjust for dark or bright shirts
    if avg_v < 60:
        s_boost = 2.3
        v_boost = 2.0
        v_offset = 50
    else:
        s_boost = 1.3
        v_boost = 1.1
        v_offset = 15

    s[mask] = np.clip(s[mask] * s_boost, 0, 255)
    v[mask] = np.clip(v[mask] * v_boost + v_offset, 0, 255)

    hsv_mod = cv2.merge([
        h.astype(np.uint8),
        s.astype(np.uint8),
        v.astype(np.uint8)
    ])
    recolored = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2RGB)

    # --- Add subtle color overlay to amplify hue visibility ---
    overlay_color = np.uint8(np.array(cv2.cvtColor(
        np.uint8([[[hue_value, 255, 255]]]), cv2.COLOR_HSV2RGB))[0][0])
    overlay = np.zeros_like(recolored)
    overlay[mask] = overlay_color

    # Blend overlay to increase color perceptibility
    recolored[mask] = cv2.addWeighted(recolored[mask], 0.65, overlay[mask], 0.35, 0)

    return recolored



def resize_keep_ratio(img, max_size=500):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def cv_to_tk(img):
    return ImageTk.PhotoImage(Image.fromarray(img))

# ================================
# GUI SETUP
# ================================
root = tk.Tk()
root.title("CS566 Project: Color Replacement")
root.geometry("1000x750")
root.configure(bg="white")

style = ttk.Style()
style.theme_use('clam')

style.configure("SaveButton.TButton", font=("Segoe UI", 11, "bold"), foreground="white",
                background="#0078D7", padding=10, relief="flat")
style.map("SaveButton.TButton", background=[("active", "#005A9E")])

style.configure("CustomColor.TButton", font=("Segoe UI", 10, "bold"), foreground="#333",
                background="#f0f0f0", padding=8, borderwidth=0)
style.map("CustomColor.TButton", background=[("active", "#d9d9d9")])

root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

header = tk.Frame(root, bg="#f8f8f8", height=70)
header.grid(row=0, column=0, sticky="ew")
tk.Label(header, text="CS566 Project: Color Replacement",
         font=("Helvetica", 22, "bold"), bg="#f8f8f8", fg="#333").pack(pady=15)

main_frame = tk.Frame(root, bg="white")
main_frame.grid(row=1, column=0, sticky="nsew")

display_img = resize_keep_ratio(shirt_img)
tk_img = cv_to_tk(display_img)
img_label = tk.Label(main_frame, image=tk_img, bg="white")
img_label.pack(pady=30)

# ================================
# COLOR SWATCHES
# ================================
color_dict = {
    'Red': (0, '#ff4d4d'),
    'Yellow': (30, '#ffd633'),
    'Green': (60, '#5cd65c'),
    'Cyan': (90, '#33cccc'),
    'Blue': (120, '#4d79ff'),
    'Magenta': (150, '#cc33ff')
}

swatch_frame = tk.Frame(main_frame, bg="white")
swatch_frame.pack(pady=15)

selected_color = tk.StringVar(value="Original")

def show_color(color_name):
    hue, hex_color = color_dict[color_name]
    recolored = recolor_shirt(hue)
    resized = resize_keep_ratio(recolored)
    tk_new = cv_to_tk(resized)
    img_label.configure(image=tk_new)
    img_label.image = tk_new
    selected_color.set(color_name)
    color_label.config(bg=hex_color)

def make_circle(canvas, color):
    r = 22
    canvas.create_oval(2, 2, 2*r, 2*r, fill=color, outline=color)

for name, (hue, hex_color) in color_dict.items():
    c = tk.Canvas(swatch_frame, width=50, height=50, bg="white", highlightthickness=0)
    make_circle(c, hex_color)
    c.bind("<Button-1>", lambda e, n=name: show_color(n))
    c.pack(side="left", padx=8)

# ================================
# FOOTER AND SAVE
# ================================
footer = tk.Frame(root, bg="#f2f2f2", height=60)
footer.grid(row=2, column=0, sticky="ew")
footer.grid_propagate(False)

footer_inner = tk.Frame(footer, bg="#f2f2f2")
footer_inner.pack(expand=True)

color_label = tk.Label(footer_inner, textvariable=selected_color,
                       font=("Arial", 16, "bold"), fg="#333", bg="#f2f2f2", width=12)
color_label.pack(side="left", padx=10)

def save_image():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    if file_path:
        hue, _ = color_dict.get(selected_color.get(), (60, ''))
        recolored = recolor_shirt(hue)
        cv2.imwrite(file_path, cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))

ttk.Button(footer_inner, text="Save Image", command=save_image,
           style="SaveButton.TButton").pack(side="left", padx=10)

root.mainloop()
