"""
Cheque Extraction Module using Traditional Computer Vision

This module provides functionality to detect and extract cheques from images,
removing the background and handling various orientations and perspectives.
"""

# Step 1. Import necessary libraries
import cv2
import numpy as np
from typing import Tuple, Optional, Literal
from pathlib import Path


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()


def find_dest(pts):
    """Find destination coordinates for perspective transform"""
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)


def scan(img):
    """Scan and extract cheque from image using contour detection and perspective transform"""
    # Keep track of resize scale for later upscaling
    dim_limit = 1080
    max_dim = max(img.shape)
    resize_scale = 1.0
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    # Create a copy of resized original image for later use
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    
    # Calculate the area of the extracted region
    extracted_width = destination_corners[2][0]
    extracted_height = destination_corners[2][1]
    extracted_area = extracted_width * extracted_height
    original_area = h * w
    
    # Check if extracted area is at least 20% of original image
    if extracted_area < (original_area * 0.2):
        # Return original image if extracted area is too small
        return orig_img
    
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_CUBIC)
    return final

def scan_faster(img, preserve_quality=True):
    """Scan and extract cheque from image using contour detection and perspective transform"""
    # Store original image for high-quality output
    original_full_res = img.copy()
    
    # Work on a smaller version for detection only
    detection_dim = 800  # Smaller for faster processing
    max_dim = max(img.shape)
    detection_scale = 1.0
    
    if max_dim > detection_dim:
        detection_scale = detection_dim / max_dim
        img = cv2.resize(img, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    
    # Create a copy of resized image for detection
    detection_img = img.copy()
    
    # Simplified morphology - reduced iterations
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # REMOVE GrabCut - it's the slowest and may not be necessary for cheque detection
    # Most cheques have clear edges even without background removal
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Smaller kernel
    
    # Edge Detection
    canny = cv2.Canny(gray, 50, 150)  # Better thresholds for documents
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    # Finding contours - use CHAIN_APPROX_SIMPLE for speed
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    if len(page) == 0:
        return original_full_res if preserve_quality else detection_img
    
    # Find the cheque contour
    corners = None
    for c in page:
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            corners = approx
            break
    
    if corners is None:
        return original_full_res if preserve_quality else detection_img
    
    # Convert corners to list and order them
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)
    
    # Scale corners back to original image dimensions if quality preservation is enabled
    if preserve_quality and detection_scale != 1.0:
        corners = [[int(x / detection_scale), int(y / detection_scale)] for x, y in corners]
        work_img = original_full_res
    else:
        work_img = detection_img
    
    destination_corners = find_dest(corners)
    
    h, w = work_img.shape[:2]
    
    # Calculate area validation
    extracted_width = destination_corners[2][0]
    extracted_height = destination_corners[2][1]
    extracted_area = extracted_width * extracted_height
    original_area = h * w
    
    if extracted_area < (original_area * 0.2):
        return original_full_res if preserve_quality else work_img
    
    # Apply perspective transform on high-resolution image
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    final = cv2.warpPerspective(work_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_CUBIC)
    return final

def extract_cheque(image_path: str, output_dir: str, scan_type : Literal["Normal", "Faster"]) -> str:
    """
    Extract and save a cheque from an image file.
    
    Args:
        image_path: Path to the input image file
        output_dir: Directory where the processed image will be saved
        
    Returns:
        Path to the saved processed image
        
    Raises:
        FileNotFoundError: If the input image file doesn't exist
        ValueError: If the image cannot be read
    """
    # Convert paths to Path objects
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    # Check if input file exists
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert from BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    if scan_type == "Normal":
        scanned_image = scan(image_rgb)
    elif scan_type == "Faster":
        scanned_image = scan_faster(image_rgb, preserve_quality=True)
    else:
        raise ValueError(f"Invalid scan_type: {scan_type}. Choose 'Normal' or 'Faster'.")
    
    # Convert back to BGR for saving with OpenCV
    scanned_image_bgr = cv2.cvtColor(scanned_image, cv2.COLOR_RGB2BGR)
    
    # Create output filename
    output_filename = f"extracted_{image_path.stem}{image_path.suffix}"
    output_path = output_dir / output_filename
    
    # Save the processed image
    cv2.imwrite(str(output_path), scanned_image_bgr)
    
    return str(output_path)

