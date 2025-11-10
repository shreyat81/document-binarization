import os
import cv2

# Input paths (your unified dataset after renaming)
images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/DIBCO_CLEAN/images"
gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/DIBCO_CLEAN/gt"

# Output paths (new grayscale versions)
out_images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/grey_scale/images"
out_gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/grey_scale/gt"

os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_gt_dir, exist_ok=True)

def convert_folder_to_gray(in_dir, out_dir):
    for fname in os.listdir(in_dir):
        if fname.startswith('.'):  # skip hidden files
            continue
        path = os.path.join(in_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Convert only if not already grayscale
        if img is None:
            continue
        if len(img.shape) == 3:  # color image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img  # already grayscale

        # Save in same extension
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, gray)

    print(f"✅ Converted {len(os.listdir(in_dir))} files in {in_dir}")

# Convert originals and GTs
convert_folder_to_gray(images_dir, out_images_dir)
convert_folder_to_gray(gt_dir, out_gt_dir)
