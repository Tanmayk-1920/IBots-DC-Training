import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def pencil_sketch(image_path, blur_kernel=21):
    '''
    Converts image to pencil sketch using OpenCV.
    
    param image_path: Path to input image
    param blur_kernel: Kernel Size (Must be Odd) for Gaussian blur (Default: 21)
    return: Tuple (original_image, pencil_sketch) or "No Image Found"
    '''
    # Step 1: Load Image
    image = cv2.imread(image_path)
    if image is None:
        return "No Image Found"
    
    # Step 2: Convert to Grayscale
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Invert Grayscale
    invert_gray = 255 - grayscale_img
    
    # Step 4: Apply Gaussian Blur
    blur_img = cv2.GaussianBlur(invert_gray, (blur_kernel, blur_kernel), 0)
    
    # Step 5: Invert Blurred Image
    invert_blur = 255 - blur_img
    
    # Step 6: Divide and Scale (with zero division protection)
    pencil_sketch = cv2.divide(grayscale_img, invert_blur, scale=256.0)
    
    # Step 7: Return
    return (image, pencil_sketch.astype(np.uint8))

def display_sketch(original_image, pencil_sketch):
    '''
    Displays original and pencil sketch side by side using matplotlib.
    
    param original_image: Original BGR image from cv2.imread()
    param pencil_sketch: Grayscale pencil sketch
    '''
    # Convert BGR to RGB for correct matplotlib display
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pencil_sketch, cmap='gray')
    axes[1].set_title("Pencil Sketch")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_sketch(image, save_path):
    '''
    Saves image to specified path.
    
    param save_path: Path where image should be saved
    param image: Image array to save (BGR format)
    '''
    if save_path is None:
        save_path = "pencil_sketch_output.jpg"
    
    success = cv2.imwrite(save_path, image)
    if success:
        print(f"Image saved successfully to: {save_path}")
    else:
        print("Error: Could not save image")

def main():
    '''
    Main function: Handles user input and orchestrates the workflow.
    '''
    # Get image path
    image_path = input("Please enter the path to image: ").strip()
    
    # Validate image exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return
    
    # Get kernel input
    while True:
        kernel_input = input("Enter kernel size (odd integer) or 'None' for default(21): ").strip()
        
        if kernel_input.lower() == 'none':
            blur_kernel = 21
            break
        
        try:
            kernel = int(kernel_input)
            if kernel % 2 == 1 and kernel > 0:  # Must be positive odd integer
                blur_kernel = kernel
                break
            else:
                print("Please enter a positive odd integer (e.g., 21, 15, 25) or 'None'")
        except ValueError:
            print("Invalid input. Please enter a number or 'None'")
    
    # Generate pencil sketch
    result = pencil_sketch(image_path, blur_kernel)
    
    if isinstance(result, str):#Checks if result is generated
        print(result)
        return
    
    original, sketch = result
    
    # Display results
    display_sketch(original, sketch)
    
    # Ask to save
    save_choice = input("Do you want to save the sketch? (Y/N): ").strip().lower()
    if save_choice == 'y':
        save_path = input("Enter save path (or press Enter for 'pencil_sketch_output.jpg'): ").strip()
        if not save_path:
            save_path = "pencil_sketch_output.jpg"
        save_sketch(sketch, save_path)

main()
