import os
import shutil
import random

# Source directories (patches)
images_dir = r"/Users/shreyatiwari/Documents/Soft Computing Project/resize_patch/images"
gt_dir     = r"/Users/shreyatiwari/Documents/Soft Computing Project/resize_patch/gt"

# Output split dataset
out_root = r"/Users/shreyatiwari/Documents/Soft Computing Project/split"
splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(out_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, split, "gt"), exist_ok=True)

# Get all image patches
all_images = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
all_images.sort()

# Shuffle for randomness
random.seed(42)
random.shuffle(all_images)

# Split sizes (70/20/10)
n_total = len(all_images)
n_train = int(0.7 * n_total)
n_test  = int(0.2 * n_total)
n_val   = n_total - n_train - n_test   # remaining 10%

train_files = all_images[:n_train]
test_files  = all_images[n_train:n_train+n_test]
val_files   = all_images[n_train+n_test:]

def copy_split(files, split_name):
    for f in files:
        # image
        shutil.copy(os.path.join(images_dir, f),
                    os.path.join(out_root, split_name, "images", f))
        # matching GT file
        base = f.replace(".npy", "")
        gt_file = base.replace("_p", "_GT_p") + ".npy"
        if os.path.exists(os.path.join(gt_dir, gt_file)):
            shutil.copy(os.path.join(gt_dir, gt_file),
                        os.path.join(out_root, split_name, "gt", gt_file))

copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

print(f"✅ Done: {n_train} train, {n_test} test, {n_val} val patches")
