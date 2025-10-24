from .json import RealPersonJsonInitializer, RealPersonJson
from .json import MarketInitializer, SYSUMM01Initializer, MARSInitializer
from .dataset import Dataset
from .mask import make_mask

import os
import enum
from PIL import Image
import torchvision.transforms as transforms


def save_img(img, dirname, filename):
    if img is None:
        img = Image.new('RGB', (128, 256), color='black')
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = os.path.join(dirname, filename)
    if isinstance(img, str):    
        img = Image.open(img)
    img.save(path)

def save_imgs(imgs, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        import shutil
        shutil.rmtree(dirname)
        os.makedirs(dirname)
    
    for i, img in enumerate(imgs):
        save_img(img, dirname, f"{i}.jpg")

def save_items(img_tgt, pose_tgt, render_tgt, imgs_ref, poses_ref, dirname):
    save_img(img_tgt, dirname, "tgt.jpg")
    save_img(pose_tgt, dirname, "pose.jpg")
    save_img(render_tgt, dirname, "render.jpg")
    save_imgs(imgs_ref, os.path.join(dirname, "imgs_ref"))
    save_imgs(poses_ref, os.path.join(dirname, "poses_ref"))

def save_items_tensor(img_tgt, bkgd_tgt, pose_tgt, imgs_ref, reids_ref, poses_ref, dirname):
    to_pil = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),
        transforms.ToPILImage(),
    ])
    imgs_ref_pil = []
    reids_ref_pil = []
    poses_ref_pil = []
    for i, img_ref in enumerate(imgs_ref):
        img_ref = to_pil(img_ref)
        imgs_ref_pil.append(img_ref)
    save_imgs(imgs_ref_pil, os.path.join(dirname, "imgs_ref"))
    for i, reid_ref in enumerate(reids_ref):
        reid_ref = to_pil(reid_ref)
        reids_ref_pil.append(reid_ref)
    save_imgs(reids_ref_pil, os.path.join(dirname, "reids_ref"))
    for i, pose_ref in enumerate(poses_ref):
        pose_ref = to_pil(pose_ref)
        poses_ref_pil.append(pose_ref)
    save_imgs(poses_ref_pil, os.path.join(dirname, "poses_ref"))
    img_tgt = to_pil(img_tgt)
    bkgd_tgt = to_pil(bkgd_tgt)
    pose_tgt = to_pil(pose_tgt)
    save_img(img_tgt, dirname, "tgt.jpg")
    save_img(pose_tgt, dirname, "pose.jpg")
    save_img(bkgd_tgt, dirname, "bkgd.jpg")
    
    



