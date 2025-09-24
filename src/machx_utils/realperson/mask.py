import os
from PIL import Image
import numpy as np
import random

def img_wobkgd(img_reid):
    img_mask = Image.new("L", img_reid.size, color=0)
    img_fegd = img_reid
    img_bkgd = Image.new("RGB", img_reid.size, color="black")
    return img_mask, img_fegd, img_bkgd


def make_mask(img_reid, img_render, rate_mask_aug = 0):
    if not img_reid:
        return None, None, None

    if isinstance(img_reid, str):
        if os.path.exists(img_reid):
            img_reid = Image.open(img_reid)
        else:
            return None, None, None
        
    if not img_render:        
        return img_wobkgd(img_reid)
    
    if isinstance(img_render, str):
        if os.path.exists(img_render):
            img_render = Image.open(img_render).convert('L')
        else:
            return img_wobkgd(img_reid)

    arr = np.array(img_render)
    h, w = arr.shape
    step_h = h // 8
    step_w = w // 8
    cache = []
    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            end_i = min(i + step_h, h)
            end_j = min(j + step_w, w)
            block = arr[i:end_i, j:end_j]
            start_i = max(0, i - step_h // 8)
            end_i = min(i + step_h + step_h // 8, h)
            start_j = max(0, j - step_w // 8)
            end_j = min(j + step_w + step_w // 8, w)
            if np.any(block > 10):
                cache.append((start_i, end_i, start_j, end_j))
            else:
                if random.random() < rate_mask_aug:
                    cache.append((start_i, end_i, start_j, end_j))
                else:
                    arr[start_i:end_i, start_j:end_j] = 0
    for (start_i, end_i, start_j, end_j) in cache:
        arr[start_i:end_i, start_j:end_j] = 1
        
    img_mask = Image.fromarray((arr > 0).astype(np.uint8) * 255)
    img_fore = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    img_back = Image.new("RGB", (w, h), (0, 0, 0))  # Black background
    pixel_reid = img_reid.load()
    pixel_fore = img_fore.load()
    pixel_back = img_back.load()
    for i in range(h):
        for j in range(w):
            if arr[i, j] > 0:  # Foreground pixel
                pixel_fore[j, i] = pixel_reid[j, i]
            else:  # Background pixel
                pixel_back[j, i] = pixel_reid[j, i]
    return img_mask, img_fore, img_back
                

    