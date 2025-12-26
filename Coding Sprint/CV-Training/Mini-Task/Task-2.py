import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1 : Load Image
image_path = input("Please enter the path to image: ").strip()

# Validate image exists
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found.")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert BGR (OpenCV) to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Creating Histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(, cmap='gray')
axes[1].set_title("Pencil Sketch")
axes[1].axis('off')    
plt.tight_layout()
plt.show()