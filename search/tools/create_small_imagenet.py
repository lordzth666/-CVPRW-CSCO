import numpy as np
import cv2
import os
from tqdm.contrib.concurrent import thread_map
import shutil

img_dir_train = "/data/ImageNetSmall/train"
img_dir_val = "/data/ImageNetSmall/val"

NUM_CLASSES = 10

def resize_images_per_folder(folder_name):
    try:
        all_files_list = os.listdir(folder_name)
    except Exception:
        return
    if len(all_files_list) == 0:
        return
    for all_files in os.listdir(folder_name):
        if not all_files.endswith(".JPEG"):
            continue
        full_img_name = os.path.join(folder_name, all_files)
        img = cv2.imread(full_img_name)
        h, w = img.shape[0], img.shape[1]
        scale_factor = min(h, w) / 256
        new_h, new_w = int(h / scale_factor), int(w / scale_factor)
        img = cv2.resize(img, dsize=(new_h, new_w), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(full_img_name, img)

def resize_images(img_dir, selected_paths):
    # First, remove the paths that are not selected.
    for path in os.listdir(img_dir):
        if path not in selected_paths and path.startswith("n"):
            shutil.rmtree(os.path.join(img_dir, path))
    # Next, carry thread map.
    all_dirs = [
        os.path.join(img_dir, path) for path in os.listdir(img_dir) if path.startswith("n")]
    thread_map(resize_images_per_folder, all_dirs, max_workers=10)

all_paths = []
for p in os.listdir(img_dir_train):
    if p.startswith("n"):
        all_paths.append(p)

# assert len(all_paths) == 1000

selected_paths = np.random.choice(all_paths, NUM_CLASSES, replace=False)
resize_images(img_dir_train, selected_paths)
resize_images(img_dir_val, selected_paths)