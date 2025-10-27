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


def save_item_tensor(img_tgt_tensor, bkgd_tgt_tensor, pose_tgt_tensor, vis_tgt, 
        img_ref_tensor, reid_ref_tensor, pose_ref_tensor, vis_ref_list, dirname_sample):
    to_pil = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),
        transforms.ToPILImage(),
    ])
    img_ref_list = []
    reid_ref_list = []
    pose_ref_list = []
    for i, img_ref in enumerate(img_ref_tensor):
        img_ref = to_pil(img_ref)
        img_ref_list.append(img_ref)
    save_imgs(img_ref_list, os.path.join(dirname_sample, "img_ref_tensor"))
    for i, reid_ref in enumerate(reid_ref_tensor):
        reid_ref = to_pil(reid_ref)
        reid_ref_list.append(reid_ref)
    save_imgs(reid_ref_list, os.path.join(dirname_sample, "reid_ref_tensor"))
    for i, pose_ref in enumerate(pose_ref_tensor):
        pose_ref = to_pil(pose_ref)
        pose_ref_list.append(pose_ref)
    save_imgs(pose_ref_list, os.path.join(dirname_sample, "pose_ref_tensor"))
    img_tgt_pil = to_pil(img_tgt_tensor[0])
    bkgd_tgt_pil = to_pil(bkgd_tgt_tensor[0])
    pose_tgt_pil = to_pil(pose_tgt_tensor[0])
    save_img(img_tgt_pil, dirname_sample, "img_tgt_tensor.jpg")
    save_img(pose_tgt_pil, dirname_sample, "pose_tgt_tensor.jpg")
    save_img(bkgd_tgt_pil, dirname_sample, "bkgd_tgt_tensor.jpg")

    # def load_items_tensor(dirname):
        
    
    
