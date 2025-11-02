from .json import RealPersonJsonInitializer, RealPersonJson
from .json import MarketInitializer, SYSUMM01Initializer, MARSInitializer
from .json import MSMT17V1Initializer, MSMT17V2Initializer, DUKEInitializer, OCCReIDInitializer
from .dataset import Dataset
from .mask import make_mask
from .style_dataset import StyleDataset

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



def save_list_with_json(list, filename):
    """
    使用JSON保存两个list到文件
    """
    data = {'list': list}
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_list_with_json(filename):
    """
    从JSON文件加载两个list
    """
    import json
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['list']



def save_item_tensor(img_tgt_tensor, bkgd_tgt_tensor, pose_tgt_tensor, 
        domain_tgt_tensor, style_tgt_tensor, vis_tgt, img_ref_tensor, 
        reid_ref_tensor, pose_ref_tensor, vis_ref_list, dirname_sample):
    to_pil = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[2]),
        transforms.ToPILImage(),
    ])
    to_pil_imgnet = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
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
    domain_tgt_pil = to_pil_imgnet(domain_tgt_tensor[0])
    style_tgt_pil = to_pil_imgnet(style_tgt_tensor[0])
    save_img(img_tgt_pil, dirname_sample, "img_tgt_tensor.jpg")
    save_img(pose_tgt_pil, dirname_sample, "pose_tgt_tensor.jpg")
    save_img(bkgd_tgt_pil, dirname_sample, "bkgd_tgt_tensor.jpg")
    save_img(domain_tgt_pil, dirname_sample, "domain_tgt_tensor.jpg")
    save_img(style_tgt_pil, dirname_sample, "style_tgt_tensor.jpg")
    save_list_with_json(vis_ref_list, os.path.join(dirname_sample, "vis_ref_list.json"))
    save_list_with_json(vis_tgt, os.path.join(dirname_sample, "vis_tgt.json"))


def load_imgs(dir_path, transform):
    """加载目录中的所有图像并转换为张量"""
    img_tensors = []
    if os.path.exists(dir_path):
        for img_file in sorted(os.listdir(dir_path)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, img_file)
                img = Image.open(img_path)
                img_tensor = transform(img)
                img_tensors.append(img_tensor)
    return img_tensors


def load_single_img(file_path, transform):
    """加载单个图像并转换为张量"""
    if os.path.exists(file_path):
        img = Image.open(file_path)
        return transform(img).unsqueeze(0)  # 添加batch维度
    return None


def load_json(file_path):
    """加载JSON文件"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


def load_item_tensor(dirname_sample):
    """
    加载保存的数据
    Args:
        dirname_sample: 保存数据的目录路径
    Returns:
        包含所有加载数据的字典
    """
    # 定义反归一化转换
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 反归一化
    ])
    
    to_tensor_imgnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )  # 反归一化
    ])
    
    # 加载参考图像列表
    img_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "img_ref_tensor"), 
        to_tensor
    )
    
    reid_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "reid_ref_tensor"), 
        to_tensor
    )
    
    pose_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "pose_ref_tensor"), 
        to_tensor
    )
    
    # 加载目标图像
    img_tgt_tensor = load_single_img(
        os.path.join(dirname_sample, "img_tgt_tensor.jpg"), 
        to_tensor
    )
    
    bkgd_tgt_tensor = load_single_img(
        os.path.join(dirname_sample, "bkgd_tgt_tensor.jpg"), 
        to_tensor
    )
    
    pose_tgt_tensor = load_single_img(
        os.path.join(dirname_sample, "pose_tgt_tensor.jpg"), 
        to_tensor
    )
    
    domain_tgt_tensor = load_single_img(
        os.path.join(dirname_sample, "domain_tgt_tensor.jpg"), 
        to_tensor_imgnet
    )
    
    style_tgt_tensor = load_single_img(
        os.path.join(dirname_sample, "style_tgt_tensor.jpg"), 
        to_tensor_imgnet
    )
    
    # 加载可见性列表
    vis_ref_list = load_json(os.path.join(dirname_sample, "vis_ref_list.json"))
    vis_tgt = load_json(os.path.join(dirname_sample, "vis_tgt.json"))
    
    # 返回与save函数参数对应的字典
    return {
        'img_tgt_tensor': img_tgt_tensor,
        'bkgd_tgt_tensor': bkgd_tgt_tensor,
        'pose_tgt_tensor': pose_tgt_tensor,
        'domain_tgt_tensor': domain_tgt_tensor,
        'style_tgt_tensor': style_tgt_tensor,
        'vis_tgt': vis_tgt,
        'img_ref_tensor': img_ref_tensor,
        'reid_ref_tensor': reid_ref_tensor,
        'pose_ref_tensor': pose_ref_tensor,
        'vis_ref_list': vis_ref_list
    }

def load_items_tensor(dirname_root):
    for dirname_sub in os.listdir(dirname_root):
        dirname = os.path.join(dirname_root, dirname_sub)
        sample = load_item_tensor(dirname)