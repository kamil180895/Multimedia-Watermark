import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def embed_watermark(frame, watermark, alpha=0.8):
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y_channel = ycrcb_frame[:, :, 0]
    dct = cv2.dct(np.float32(Y_channel))
    resized_watermark = cv2.resize(watermark, (Y_channel.shape[1], Y_channel.shape[0]))
    
    # Normalizacja watermarku
    resized_watermark = cv2.normalize(resized_watermark, None, 0, 1, cv2.NORM_MINMAX)
    
    # Debugowanie DCT
    print("DCT przed watermarkiem:", dct)
    dct += alpha * resized_watermark
    print("DCT po dodaniu watermarku:", dct)
    
    Y_channel_watermarked = cv2.idct(dct)
    Y_channel_watermarked = np.uint8(np.clip(Y_channel_watermarked, 0, 255))
    ycrcb_frame[:, :, 0] = Y_channel_watermarked
    watermarked_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)
    return watermarked_frame

def extract_watermark(frame, original_frame, watermark_shape, alpha=0.8):
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YCrCb)
    Y_watermarked = np.float32(ycrcb_frame[:, :, 0])
    Y_original = np.float32(ycrcb_original_frame[:, :, 0])
    dct_watermarked = cv2.dct(Y_watermarked)
    dct_original = cv2.dct(Y_original)
    extracted_watermark = (dct_watermarked - dct_original) / alpha
    extracted_watermark = np.uint8(np.clip(extracted_watermark, 0, 255))
    return cv2.resize(extracted_watermark, (watermark_shape[1], watermark_shape[0]))

def calculate_psnr(original_frame, compressed_frame):
    mse = np.mean((original_frame - compressed_frame) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original_frame, compressed_frame):
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    compressed_gray = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, compressed_gray)

def process_video(input_video_path, watermark_path, output_video_path, output_extracted_watermark, alpha=0.1):
    cap = cv2.VideoCapture(input_video_path)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        raise ValueError("Failed to load the watermark.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    watermark = cv2.resize(watermark, (frame_width, frame_height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_watermarked = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    psnr_values = []
    ssim_values = []
    cumulative_watermark = np.zeros_like(watermark, dtype=np.float32)
    frame_count = 0

    first_frame_original = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame_original is None:
            first_frame_original = frame.copy()
        
        watermarked_frame = embed_watermark(frame, watermark, alpha)
        out_watermarked.write(watermarked_frame)

        extracted_watermark = extract_watermark(watermarked_frame, first_frame_original, watermark.shape, alpha)
        cumulative_watermark += extracted_watermark
        frame_count += 1

        psnr_value = calculate_psnr(frame, watermarked_frame)
        ssim_value = calculate_ssim(frame, watermarked_frame)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    cap.release()
    out_watermarked.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_watermark = cumulative_watermark / frame_count
        cv2.imwrite(output_extracted_watermark, np.uint8(np.clip(avg_watermark, 0, 255)))

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

# Ścieżki do plików
input_video_path = '../Downloads/a2560x1440_2K.mp4'
watermark_path = '../Downloads/watermark.png'
output_video_path = '../Downloads/watermarked_video.mp4'
output_extracted_watermark = '../Downloads/extracted_watermark.png'

process_video(input_video_path, watermark_path, output_video_path, output_extracted_watermark)
