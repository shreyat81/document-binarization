import os
import cv2
import numpy as np

# Input (your grayscale dataset)
images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/grey_scale/images"
gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/grey_scale/gt"

# Output normalized arrays (saved as .npy for ML training)
out_images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/normalization/images"
out_gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/normalization/gt"

os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_gt_dir, exist_ok=True)

def normalize_and_save(in_dir, out_dir, is_gt=False):
    for fname in os.listdir(in_dir):
        if fname.startswith("."):
            continue
        path = os.path.join(in_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        if is_gt:
            # GT are binary images → keep as 0 and 1
            norm = (img > 127).astype(np.uint8)
        else:
            # Originals → normalize to [0,1] float32
            norm = img.astype(np.float32) / 255.0

        # Save as numpy array
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, base + ".npy")
        np.save(out_path, norm)

    print(f"✅ Normalized {len(os.listdir(in_dir))} files from {in_dir}")

# Normalize originals and GT
normalize_and_save(images_dir, out_images_dir, is_gt=False)
normalize_and_save(gt_dir, out_gt_dir, is_gt=True)
