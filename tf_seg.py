# libraries needed
import cv2 as cv
import numpy as np
from util_func import *
import argparse
from skimage.exposure import is_low_contrast
import time

# Function to enhance the contrast of image
def contrast_enhance(img):
    """Args:
    img: BRG image array
    
    output: enhanced image array"""
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    L, a, b = cv.split(img_lab)
    L = cv.equalizeHist(L)
    img_lab_merge = cv.merge((L, a, b))
    return cv.cvtColor(img_lab_merge, cv.COLOR_Lab2BGR)

# Function to perform zero parameter Canny edge detection
def auto_canny(img, method, sigma=0.33):
    """
    Args:
    img: grayscale image
    method: Otsu, triangle, and median
    sigma: 0.33 (default)
    2 outputs:
    edge_detection output, the high threshold (to be used as input argument for Hough Circular Transform)"""
    if method=="median":
        Th = np.median(img)
        
    elif method=="triangle":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE)
        
    elif method=="otsu":
        Th, _ = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
        
    else:
        raise Exception("method specified not available!")
        
    lowTh = (1-sigma) * Th
    highTh = (1+sigma) * Th
    
    return cv.Canny(img, lowTh, highTh), highTh

# Function to perform color based segmentation
def color_seg(img, kernel_size=None):
    """Args:
    img: image in bgr
    kernel_size: None (default:(3, 3))"""
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mask_red1 = cv.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_img, lower_red2, upper_red2)
    mask_blue = cv.inRange(hsv_img, lower_blue, upper_blue)
    mask_yellow = cv.inRange(hsv_img, lower_yellow, upper_yellow)
    mask_black = cv.inRange(hsv_img, lower_black, upper_black)
    
    mask_combined = mask_red1 | mask_red2 | mask_blue | mask_yellow | mask_black
    
    if kernel_size is not None:
        kernel = np.ones(kernel_size, np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)
        
    mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_OPEN, kernel)
    mask_combined = cv.morphologyEx(mask_combined, cv.MORPH_CLOSE, kernel)
    
    return mask_combined

# rectangle detection (using Douglas-Peuker algorithm)
def cnt_rect(cnts, coef=0.1):
    contour_list = []
    for cnt in cnts:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, coef*peri, True)
        if len(approx) == 4:
            contour_list.append(cnt)

    if not contour_list:
        return None
    else:
        LC = max(contour_list, key=cv.contourArea)
        return LC

# circle detection
def cnt_circle(img, hough_dict):
    """Args:
    img: Grayscale Image after resizing
    hough_dict: hough_circle_transform parameters"""
    mask = np.zeros_like(img)
    circles = cv.HoughCircles(img, 
                              cv.HOUGH_GRADIENT, 
                              hough_dict["dp"], 
                              hough_dict["minDist"], 
                              param1=hough_dict["param1"], 
                              param2=hough_dict["param2"],
                              minRadius=hough_dict["minRadius"], 
                              maxRadius=hough_dict["maxRadius"])
    if circles is None:
        return circles
    else:
        # perform LCA
        list_circles = circles[0]
        largest_circles = max(list_circles, key=lambda x: x[2])
        center_x, center_y, r = largest_circles
        cv.circle(mask, (int(center_x), int(center_y)), int(r), 255)
        cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]
        if len(cnts[0])>0:
            return max(cnt, key=cv.contourArea)
        else:
            return cnt[-1]

# combine the results of 2 shape detectors
def integrate_circle_rect(rect_cnt, circle_cnt, cnt):
    if circle_cnt is not None and rect_cnt is not None:
        # compare the area
        if cv.contourArea(circle_cnt) >= cv.contourArea(rect_cnt):
            output = circle_cnt
        else:
            output = rect_cnt

    elif circle_cnt is not None and rect_cnt is None:
        output = circle_cnt

    elif circle_cnt is None and rect_cnt is not None:
        output = rect_cnt

    else:
        if len(cnt)==0:
            return np.array([])
        else:
            output = max(cnt, key=cv.contourArea)

    return output

# combine the results of edge detector + color based segmentation followed by shape detection combined results
def integrate_edge_color(output1, output2):
    if not isinstance(output1, np.ndarray):
        output1 = np.array(output1)
        
    if not isinstance(output2, np.ndarray):
        output2 = np.array(output2)
        
    if len(output1)==0 and len(output2)==0:
        return np.array([])
    
    elif len(output1)==0 and output2.shape[-1]==2:
        return output2
    
    elif len(output2)==0 and output1.shape[-1]==2:
        return output1
    
    else:
        if cv.contourArea(output1[0]) > cv.contourArea(output2[0]):
            return output1
        else:
            return output2

# circle detection (Hough circle transform parameter settings)
hough_circle_parameters = {
    "dp": 1,
    "minDist": 150,
    "param1": 200,    # adaptively change according to image
    "param2": 15,  
    "minRadius": 10,
    "maxRadius": 100
}

# Color segmentation ranges of HSV color spaces (red, blue, yellow and black)
lower_red1 = (0, 40, 50)
upper_red1 = (10, 255, 210)
lower_red2 = (165, 40, 50)
upper_red2 = (179, 255, 210)

# Blue color 
lower_blue = (90, 40, 50)
upper_blue = (120, 255, 210)

# Yellow colors
lower_yellow = (20, 40, 50)
upper_yellow = (35, 255, 210)

# black colors
lower_black = (0, 0, 0)
upper_black = (179, 255, 5)

# width after resize
fixed_width = 200

# Main operations
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fn", required=True, 
                help="filename")
args = vars(ap.parse_args())

# Operation start
# Read image
start_time = time.time()
img = cv.imread(cv.samples.findFile(args["fn"]))
img_copy = img.copy()
# Preprocess
img_denoised = cv.medianBlur(img_copy, 3)
if is_low_contrast(img_denoised):
    img_denoised = contrast_enhance(img_denoised)
# Resize the image
ratio = fixed_width / img.shape[1]
img_resized = cv.resize(img_denoised, None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)

# change to grayscale
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

#1: Edge detection + shape detection + combine results of shape detector
edge, canny_th2 = auto_canny(gray, "otsu")

# Perform shape detectors
cnts = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = cnts[0]
rect = cnt_rect(cnt)

hough_circle_parameters["param1"] = canny_th2
circle = cnt_circle(gray, hough_circle_parameters)

output1 = integrate_circle_rect(rect, circle, cnt)

# color segmentation
color_segmented = color_seg(img_resized)

# perform rectangular object detection
cnts = cv.findContours(color_segmented, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnt = cnts[0]
rect = cnt_rect(cnt)

# perform circular object detection
hough_circle_parameters["param1"] = 200
circle = cnt_circle(color_segmented, hough_circle_parameters)

output2 = integrate_circle_rect(rect, circle, cnt)

# integrate output1 and output2
final_output = integrate_edge_color(output1, output2)

# Take the execution time
print(f"The execution time of this pipeline: {(time.time()-start_time):.3f}s")

if len(final_output) == 0:
    print("no detection!")
    show_img("no detection", img_resized)
else:
    x, y, w, h = cv.boundingRect(final_output)
    cv.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
    show_img("results", img_resized)