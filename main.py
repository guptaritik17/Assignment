# Adaptive Pipeline

import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

def load_dicom_file(file_path):
    """Safely load a DICOM or DICOM-compatible RVG image file."""
    try:
        ds = pydicom.dcmread(file_path, force=True)
        img = ds.pixel_array.astype('float32')

        # Apply rescale slope and intercept if present
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        img = img * slope + intercept

        # Invert MONOCHROME1 if needed
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = img.max() - img

        # Normalize image to 8-bit
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return img.astype('uint8'), ds

    except Exception as e:
        print(f" Error loading {file_path}: {str(e)}")
        return None, None

def adaptive_preprocessing(img):
    """Run adaptive preprocessing on an image."""
    try:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Estimate image contrast and sharpness
        contrast = np.percentile(img, 99) - np.percentile(img, 1)
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

        # Estimate noise from 8 patches
        h, w = img.shape
        patch_size = 40
        patches = [
            img[0:patch_size, 0:patch_size],
            img[0:patch_size, w-patch_size:w],
            img[h-patch_size:h, 0:patch_size],
            img[h-patch_size:h, w-patch_size:w],
            img[h//2-patch_size//2:h//2+patch_size//2, 0:patch_size],
            img[h//2-patch_size//2:h//2+patch_size//2, w-patch_size:w],
            img[0:patch_size, w//2-patch_size//2:w//2+patch_size//2],
            img[h-patch_size:h, w//2-patch_size//2:w//2+patch_size//2]
        ]
        noise_level = np.mean([np.std(p) for p in patches])

        # Apply CLAHE
        clahe_clip = np.clip(contrast / 40, 1.0, 3.0)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)

        # Apply bilateral filter for edge-preserving smoothing
        img_bilateral = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

        # Denoise using fastNlMeansDenoising
        denoise_strength = np.clip(noise_level * 0.6, 5, 15)
        img_denoised = cv2.fastNlMeansDenoising(
            img_bilateral, None,
            h=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # Adaptive sharpening using weighted Gaussian blur
        if sharpness < 20:
            strength = 2.0
        elif sharpness < 80:
            strength = 1.7
        elif sharpness > 300:
            strength = 0.7
        else:
            strength = 1.1

        gaussian = cv2.GaussianBlur(img_denoised, (5, 5), 1.2)
        img_sharpened = cv2.addWeighted(img_denoised, strength, gaussian, -(strength - 1), 0)

        return img_sharpened

    except Exception as e:
        print(f" Adaptive preprocessing failed: {str(e)}")
        return None

def process_file(file_path):
    """Process a single image file through the adaptive pipeline."""
    print(f"\n=== Processing {file_path} ===")
    img, ds = load_dicom_file(file_path)
    if img is None:
        return

    result = adaptive_preprocessing(img)
    if result is None:
        return

    # Save result image next to original
    out_path = os.path.splitext(file_path)[0] + "_processed.png"
    try:
        cv2.imwrite(out_path, result)
        print(f"✅ Saved processed image: {out_path}")
    except Exception as e:
        print(f" Could not save processed image: {str(e)}")

if __name__ == "__main__":
    image_dir = "Enter your Path"  # directory path
    try:
        sample_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                        if f.lower().endswith(('.dcm', '.rvg'))]
    except Exception as e:
        print(f" Failed to list directory '{image_dir}': {str(e)}")
        sample_files = []

    for file in sample_files:
        if os.path.exists(file):
            process_file(file)
        else:
            print(f"⚠️ File not found: {file}")
