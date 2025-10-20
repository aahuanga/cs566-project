import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser
from PIL import Image, ImageTk

# ================================
# INITIAL SETTINGS
# ================================
input_image = 'shirt_extracted.jpg'
color_dict = {
    'Red': (0, '#ff4d4d'),
    'Yellow': (30, '#ffd633'),
    'Green': (60, '#5cd65c'),
    'Cyan': (90, '#33cccc'),
    'Blue': (120, '#4d79ff'),
    'Magenta': (150, '#cc33ff')
}

# ================================
# IMAGE LOADING AND PREPARATION
# ================================
shirt_img = cv2.imread(input_image)
if shirt_img is None:
    raise FileNotFoundError(f"Could not load image: {input_image}")

shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)
mask = np.any(shirt_img != [0, 0, 0], axis=2)
hsv_base = cv2.cvtColor(shirt_img, cv2.COLOR_RGB2HSV)

def recolor_shirt(hue_value):
    hsv = hsv_base.copy()
    hsv[..., 0][mask] = hue_value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def resize_keep_ratio(img, max_size=500):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def cv_to_tk(img):
    return ImageTk.PhotoImage(Image.fromarray(img))

# ================================
# MAIN WINDOW SETUP
# ================================
root = tk.Tk()
root.title("CS566 Project: Color Replacement")
root.geometry("1000x750")
root.configure(bg="white")

# Style configuration
style = ttk.Style()
style.theme_use('clam')

# Save button styling
style.configure(
    "SaveButton.TButton",
    font=("Segoe UI", 11, "bold"),
    foreground="white",
    background="#0078D7",
    padding=10,
    relief="flat"
)
style.map(
    "SaveButton.TButton",
    background=[("active", "#005A9E")]
)

# Custom color button styling
style.configure(
    "CustomColor.TButton",
    font=("Segoe UI", 10, "bold"),
    foreground="#333",
    background="#f0f0f0",
    padding=8,
    borderwidth=0
)
style.map(
    "CustomColor.TButton",
    background=[("active", "#d9d9d9")]
)

# Layout
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# Header
header = tk.Frame(root, bg="#f8f8f8", height=70)
header.grid(row=0, column=0, sticky="ew")
tk.Label(
    header, text="CS566 Project: Color Replacement",
    font=("Helvetica", 22, "bold"), bg="#f8f8f8", fg="#333"
).pack(pady=15)

# Main content
main_frame = tk.Frame(root, bg="white")
main_frame.grid(row=1, column=0, sticky="nsew")

display_img = resize_keep_ratio(shirt_img)
tk_img = cv_to_tk(display_img)
img_label = tk.Label(main_frame, image=tk_img, bg="white")
img_label.pack(pady=30)

# Color swatches
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

# Custom color picker
# def pick_custom_color():
#     color = colorchooser.askcolor(title="Choose a custom color")
#     if not color[0]:
#         return
#     r, g, b = color[0]
#     rgb = np.uint8([[[r, g, b]]])
#     hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
#     hue_value = hsv[0][0][0]
#     recolored = recolor_shirt(hue_value)
#     resized = resize_keep_ratio(recolored)
#     tk_new = cv_to_tk(resized)
#     img_label.configure(image=tk_new)
#     img_label.image = tk_new
#     selected_color.set("Custom")
#     color_label.config(bg=color[1])

# ttk.Button(
#     swatch_frame,
#     text="Custom Color",
#     command=pick_custom_color,
#     style="CustomColor.TButton"
# ).pack(side="left", padx=12)

# Footer
footer = tk.Frame(root, bg="#f2f2f2", height=60)
footer.grid(row=2, column=0, sticky="ew")
footer.grid_propagate(False)

footer_inner = tk.Frame(footer, bg="#f2f2f2")
footer_inner.pack(expand=True)

color_label = tk.Label(
    footer_inner, textvariable=selected_color,
    font=("Arial", 16, "bold"), fg="#333", bg="#f2f2f2", width=12
)
color_label.pack(side="left", padx=10)

# Save image function
def save_image():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
    )
    if file_path:
        hue, _ = color_dict.get(selected_color.get(), (60, ''))
        recolored = recolor_shirt(hue)
        cv2.imwrite(file_path, cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))

ttk.Button(
    footer_inner,
    text="Save Image",
    command=save_image,
    style="SaveButton.TButton"
).pack(side="left", padx=10)

root.mainloop()
