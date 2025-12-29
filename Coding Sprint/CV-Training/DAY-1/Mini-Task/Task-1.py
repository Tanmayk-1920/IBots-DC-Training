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

# Step 2 : Gaussian Blur
gaussian_image = cv2.GaussianBlur(image, (7, 7), 0)

# Step 3 : Otsu thresholding (on grayscale)
gray = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2GRAY)
high_thresh_value, edges_otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

low_thresh_value = 0.5 * high_thresh_value

# Canny Edge Detection
canny_edges = cv2.Canny(gray, low_thresh_value, high_thresh_value)

# Step 4 : Binary Thresholding with average threshold
avg_thresh_value = 0.5 * (high_thresh_value + low_thresh_value)
ret, thresh_image = cv2.threshold(gray, avg_thresh_value, 255, cv2.THRESH_BINARY)

# Display 4 images in 2x2 grid
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(thresh_image, cmap="gray")
plt.title("Binary Threshold")
plt.axis("off")

plt.tight_layout()
plt.show()



