import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1 : Load Image
image_path = input("Please enter the path to image: ").strip()    
# Validate image exists
if not os.path.exists(image_path):
    print(f"Error: The file '{image_path}' was not found.")

image = cv2.imread(image_path)
if image is None:
    print("File Not Found")

# Step 2 : Gaussian Blur
gaussian_image = cv2.GaussianBlur(image , (7,7) , 0)

# Step 3 : 