import numpy as np
import cv2

def embed_watermark(frame, watermark, alpha=0.1):
    # Convert the frame to YCrCb color space
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Extract the luminance (Y) component
    Y_channel = ycrcb_frame[:, :, 0]
    
    # Apply DCT to the Y component
    dct = cv2.dct(np.float32(Y_channel))
    
    # Get the dimensions of the frame and watermark
    rows, cols = Y_channel.shape
    wm_rows, wm_cols = watermark.shape
    
    # Check if the watermark is larger than the video frame
    if wm_rows > rows or wm_cols > cols:
        raise ValueError("The watermark is larger than the video frame.")
    
    # Embed the watermark into selected DCT coefficients
    dct[0:wm_rows, 0:wm_cols] += alpha * watermark
    
    # Apply inverse DCT
    Y_channel_watermarked = cv2.idct(dct)
    
    # Normalize pixel values
    Y_channel_watermarked = np.uint8(np.clip(Y_channel_watermarked, 0, 255))
    
    # Replace the original Y component with the modified version
    ycrcb_frame[:, :, 0] = Y_channel_watermarked
    # Convert back to BGR color space
    watermarked_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)
    
    return watermarked_frame

def process_video(input_video_path, watermark_path, output_video_path, alpha=0.1):
    # Load the video and watermark
    cap = cv2.VideoCapture(input_video_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    
    if watermark is None:
        raise ValueError("Failed to load the watermark.")
    
    # Resize the watermark to appropriate dimensions
    wm_rows, wm_cols = watermark.shape
    watermark = cv2.resize(watermark, (wm_cols, wm_rows))  # Example size
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Embed the watermark into the frame
        watermarked_frame = embed_watermark(frame, watermark, alpha)
        # Write the modified frame
        out.write(watermarked_frame)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
input_video_path = '../../Downloads/sample_video.mp4'
watermark_path = '../../Downloads/sample_watermark.png'
output_video_path = '../../Downloads/hello.mp4'
process_video(input_video_path, watermark_path, output_video_path)
