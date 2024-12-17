import os
import dwt
import psnr_and_ssim


def list_absolute_filepaths(directory_path):
    """List absolute file paths of all files and subdirectories in the given directory."""
    # Convert the relative path to an absolute path
    directory_path = os.path.abspath(directory_path)
    try:
        # Get the list of all files and subdirectories in the specified path
        filenames = os.listdir(directory_path)
        absolute_paths = [os.path.join(directory_path, filename) for filename in filenames]
        return absolute_paths
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory {directory_path}.")
        return []


directory = './OriginalVideoSamples'
filepaths = list_absolute_filepaths(directory)
#make sure that your method returns watermarked video filepath
watermark_methods = [dwt.apply_watermark]


if filepaths:
    for filepath in filepaths:
        print(filepath)
        for watermark_method in watermark_methods:
            watermarked_video_filepath = watermark_method(filepath)
            average_psnr, average_ssim = psnr_and_ssim.calculate_metrics(filepath, watermarked_video_filepath)
            print(f"Average PSNR: {average_psnr:.2f}")
            print(f"Average SSIM: {average_ssim:.4f}")
        
else:
    print(f"No files found in the directory '{directory}'.")