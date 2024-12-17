import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original_frame, compressed_frame):
    """Calculate PSNR between two frames."""
    mse = np.mean((original_frame - compressed_frame) ** 2)
    if mse == 0:  # If MSE is zero, return a high PSNR value
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(original_frame, compressed_frame):
    """Calculate SSIM between two frames."""
    # Convert to grayscale for SSIM calculation
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(original_gray, compressed_gray)
    return ssim_value

def calculate_metrics(original_video_path, compressed_video_path):
    """Calculate average PSNR and SSIM for two videos."""
    original_cap = cv2.VideoCapture(original_video_path)
    compressed_cap = cv2.VideoCapture(compressed_video_path)
    
    psnr_list = []
    ssim_list = []

    while True:
        ret1, original_frame = original_cap.read()
        ret2, compressed_frame = compressed_cap.read()
        
        if not ret1 or not ret2:
            break

        psnr_value = calculate_psnr(original_frame, compressed_frame)
        ssim_value = calculate_ssim(original_frame, compressed_frame)

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

    original_cap.release()
    compressed_cap.release()

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    return avg_psnr, avg_ssim