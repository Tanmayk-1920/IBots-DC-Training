import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def pre_process(image_path):
    '''
    Docstring for pre_process
    
    :param image_path: Contains path for Image
    '''
    # Step 1: Load Image
    image = cv2.imread(image_path)
    if image is None:
        return(None , None)


    # Step 2: Convert to Grayscale
    grayscale_img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur
    blur_img = cv2.GaussianBlur(grayscale_img , (21,21) , 0)

    # Step 4: Enhance with Hist Equalization
    enhance_img = cv2.equalizeHist(blur_img)

    return (image , enhance_img)

def detect_circles(gray_img , dp = 1 , minDist = 50 ,
                    param1 = 50 , param2 = 30 , 
                    minRadius = 10 , maxRadius = 100):
    '''
    Docstring for detect_circles
    
    :param gray_img: Preprocessed Image
    :param dp: Inverse Accumulator Resolution Ratio
    :param minDist: Min Distance Between Circle Centers
    :param param1: Upper Canny Threshold
    :param param2: Lower Canny Threshold
    :param minRadius: Min Circle Radius
    :param maxRadius: Max Circle Radius
    '''
    circles = cv2.HoughCircles(gray_img , dp =dp , minDist=minDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    return circles

def visualize_circle(img, circles, save_path='hough_circles_detected.png'):
    '''
    Visualize and save detected circles.
    
    :param img: Original Color Image
    :param circles: nd array returned by HoughCircles
    :param save_path: Path to save the output image (default: 'hough_circles_detected.png')
    '''
    original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_hough = original_img_rgb.copy()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axes[0].imshow(original_img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_rgb_hough)
    axes[1].set_title("Detected Circles with Labels")
    axes[1].axis('off')

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, (x, y, r) in enumerate(circles[0, :]):
            # Green outline
            circ_outline = patches.Circle((x, y), r, edgecolor='green', facecolor='none', linewidth=2)
            axes[1].add_patch(circ_outline)
            
            # Red center
            center_dot = patches.Circle((x, y), radius=2, color='red')
            axes[1].add_patch(center_dot)
            
            # Label
            label = f"ID:{i}\nR:{r}"
            axes[1].text(x, y - r - 10, label, color='white', weight='bold', 
                         fontsize=8, ha='center', bbox=dict(facecolor='blue', alpha=0.6, pad=1))

        axes[1].set_xlim(0, img_rgb_hough.shape[1])
        axes[1].set_ylim(img_rgb_hough.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-res PNG
    print(f"Image saved to: {save_path}")
    plt.show()

def calculate_statistics(circles):
    '''
    Docstring for calculate_statistics
    
    :param circles: nd array of hough circles
    '''
    num_circles = len(circles[0])
    max_rad = np.max(circles[0, :, 2])
    min_rad = np.min(circles[0, :, 2])
    avg_rad = np.mean(circles[0, :, 2])
    
    circle_data = [{"x": float(c[0]), "y": float(c[1]), "radius": float(c[2])} for c in circles[0, :]]
    
    stats = {
        'circles': circle_data,
        'Number of Circles': num_circles,
        'Minimum Radius': min_rad,
        'Maximum Radius': max_rad,
        'Average Radius': avg_rad
    }
    return stats

def main(image_path, save_path='hough_circles_detected.png'):
    '''
    Main function to run circle detection pipeline with save capability.
    
    :param image_path: Path to the input image file
    :param save_path: Path to save the output image with detected circles
    '''
    # Preprocess image
    image, enhanced = pre_process(image_path)
    if image is None:
        print("Error: Could not load image at", image_path)
        return
    
    # Detect circles
    circles = detect_circles(enhanced)
    
    if circles is not None:
        # Visualize and save results
        visualize_circle(image, circles, save_path)
        
        # Calculate and print statistics
        stats = calculate_statistics(circles)
        print("\nCircle Detection Statistics:")
        print(f"Number of Circles: {stats['Number of Circles']}")
        print(f"Minimum Radius: {stats['Minimum Radius']:.2f}")
        print(f"Maximum Radius: {stats['Maximum Radius']:.2f}")
        print(f"Average Radius: {stats['Average Radius']:.2f}")
        print("\nIndividual Circles:")
        for i, circle in enumerate(stats['circles']):
            print(f"  Circle {i}: x={circle['x']:.1f}, y={circle['y']:.1f}, r={circle['radius']:.1f}")
        print(f"\nVisualization saved to: {save_path}")
    else:
        print("No circles detected.")

# Usage examples
if __name__ == "__main__":
    # Basic usage with default save name
    main("circles_image.jpg")
    
    # Custom save path
    main("coin_test1.jpg", "detected_coins.jpg")
    
   
