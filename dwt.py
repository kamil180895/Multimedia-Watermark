import cv2
import numpy as np
import pywt

def dwt2(image):
    """
    Perform a 2D Discrete Wavelet Transform on an image.
    """
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)

def idwt2(LL, LH, HL, HH):
    """
    Perform the inverse 2D Discrete Wavelet Transform on coefficients.
    """
    coeffs2 = LL, (LH, HL, HH)
    return pywt.idwt2(coeffs2, 'haar')

def embed_watermark_frame(frame, watermark, alpha=0.2):
    """
    Embed a watermark into a video frame using DWT.
    """
    LL, (LH, HL, HH) = dwt2(frame)
    watermark_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]))
    LL_watermarked = LL + alpha * watermark_resized
    watermarked_frame = idwt2(LL_watermarked, LH, HL, HH)
    return np.clip(watermarked_frame, 0, 255).astype(np.uint8)

def extract_watermark_frame(watermarked_frame, original_frame, alpha=0.2):
    """
    Extract the watermark from a video frame using DWT.
    """
    LL_watermarked, _ = dwt2(watermarked_frame)
    LL_original, _ = dwt2(original_frame)
    extracted_watermark = (LL_watermarked - LL_original) / alpha
    return np.clip(extracted_watermark, 0, 255).astype(np.uint8)


#Extraction Example
def extract_watermark_from_video(original_video_path, watermarked_video_path, alpha=0.2):
    """
    Extract the watermark from a watermarked video.
    """
    original_cap = cv2.VideoCapture(original_video_path)
    watermarked_cap = cv2.VideoCapture(watermarked_video_path)
    watermark = None

    print("Extracting watermark...")
    while True:
        ret1, original_frame = original_cap.read()
        ret2, watermarked_frame = watermarked_cap.read()

        if not ret1 or not ret2:
            break

        # Convert frames to grayscale
        gray_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        gray_watermarked = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2GRAY)

        # Extract watermark from the frame
        extracted_watermark = extract_watermark_frame(gray_watermarked, gray_original, alpha=alpha)

        if watermark is None:
            watermark = extracted_watermark
        else:
            watermark = np.maximum(watermark, extracted_watermark)

    original_cap.release()
    watermarked_cap.release()

    # Save the extracted watermark
    cv2.imwrite("extracted_watermark.png", watermark)
    print("Extracted watermark saved as extracted_watermark.png")

def apply_watermark(filepath):
    alpha_parameter = 0.2
    # Paths for the input and output
    video_path = filepath
    watermark_path = "watermark.png"
    output_path = "watermarked_video.mp4"

    # Load the watermark
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    print("Processing video...")
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Embed the watermark into the frame
        watermarked_frame = embed_watermark_frame(gray_frame, watermark, alpha_parameter)

        # Write the frame to the output video
        out.write(watermarked_frame)

    cap.release()
    out.release()
    print("Watermarked video saved to:", output_path)
    return output_path
