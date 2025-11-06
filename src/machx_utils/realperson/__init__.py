from .json import RealPersonJsonInitializer, RealPersonJson
from .json import MarketInitializer, SYSUMM01Initializer, MARSInitializer
from .json import MSMT17V1Initializer, MSMT17V2Initializer, DUKEInitializer, OCCReIDInitializer
from .dataset import Dataset
from .mask import make_mask
from .style_dataset import StyleDataset

import os
import enum
from PIL import Image
import torch
import torchvision.transforms as transforms


def check_ext(filename, is_img = False):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    if is_img:
        return any(filename.lower().endswith(ext) for ext in image_extensions)


def save_img(img, dirname, filename):
    if img is None:
        img = Image.new('RGB', (128, 256), color='black')
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    path = os.path.join(dirname, filename)
    if isinstance(img, str):
        if os.path.exists(img):
            img = Image.open(img)
        else:
            img = Image.new('RGB', (128, 256), color='black')
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

        
def load_single_img(file_path, transform):
    """加载单个图像并转换为张量"""
    if os.path.exists(file_path):
        img = Image.open(file_path)
        return transform(img).unsqueeze(0)  # 添加batch维度
    return None


def load_imgs_bkgd(dirname_tgt, dirname_render, transform):
    img_tensors = []
    if os.path.exists(dirname_tgt) and os.path.exists(dirname_render):
        for filename in sorted(os.listdir(dirname_tgt)):
            if not check_ext(filename, is_img=True):
                continue
            path_tgt = os.path.join(dirname_tgt, filename)
            path_render = os.path.join(dirname_render, filename)
            _, _, img_bkgd = make_mask(path_tgt, path_render)
            img_tensor = transform(img_bkgd)
            img_tensors.append(img_tensor)
    img_tensor = torch.stack(img_tensors, dim = 0)
    return img_tensor


def load_imgs(dir_path, transform):
    """加载目录中的所有图像并转换为张量"""
    img_tensors = []
    if os.path.exists(dir_path):
        for img_file in sorted(os.listdir(dir_path)):
            if check_ext(img_file, is_img=True):
                img_path = os.path.join(dir_path, img_file)
                img = Image.open(img_path)
                img_tensor = transform(img)
                img_tensors.append(img_tensor)
    img_tensor = torch.stack(img_tensors, dim = 0)
    return img_tensor


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
    

def save_sample(img_tgt_list, render_tgt_list, vis_tgt, 
            img_ref_list, pose_ref_list, vis_ref_list, dirname_root):
    save_imgs(img_ref_list, os.path.join(dirname_root, "img_ref"))
    save_imgs(pose_ref_list, os.path.join(dirname_root, "pose_ref"))
    save_imgs(img_tgt_list, os.path.join(dirname_root, "img_tgt"))
    save_imgs(render_tgt_list, os.path.join(dirname_root, "render_tgt"))
    save_list_with_json(vis_ref_list, os.path.join(dirname_root, "vis_ref_list.json"))
    save_list_with_json(vis_tgt, os.path.join(dirname_root, "vis_tgt.json"))

    
def load_sample(dirname_sample):
    from .dataset import TransformsSet
    transforms_set = TransformsSet(
        img_size=(512, 512), 
        width_scale=(1, 1), 
        height_scale=(1, 1), 
        rate_random_erase=0
    )
    
    img_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "img_ref"), 
        transforms_set("norm")
    )
    reid_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "img_ref"),
        transforms_set("reid")
    )
    pose_ref_tensor = load_imgs(
        os.path.join(dirname_sample, "pose_ref"),
        transforms_set("norm")
    )
    vis_ref_list = load_list_with_json(os.path.join(dirname_sample, "vis_ref_list.json"))

    bkgd_tgt_tensor = load_imgs_bkgd(
        os.path.join(dirname_sample, "img_tgt"),
        os.path.join(dirname_sample, "render_tgt"),
        transforms_set("norm")
    )
    domain_tgt_tensor = load_imgs(
        os.path.join(dirname_sample, "img_tgt"),
        transforms_set("domain")
    )
    style_tgt_tensor = load_imgs(
        os.path.join(dirname_sample, "img_tgt"),
        transforms_set("style")
    )
    vis_tgt = load_list_with_json(os.path.join(dirname_sample, "vis_tgt.json"))

    personid = dirname_sample.split("/")[-1]
    
    return {
        'img_ref_tensor': img_ref_tensor,
        'reid_ref_tensor': reid_ref_tensor,
        'pose_ref_tensor': pose_ref_tensor,
        'vis_ref_list': vis_ref_list,
        'bkgd_tgt_tensor': bkgd_tgt_tensor,
        'domain_tgt_tensor': domain_tgt_tensor,
        'style_tgt_tensor': style_tgt_tensor,
        'vis_tgt': vis_tgt,
        'personid': personid,
        'dirname': dirname_sample,
    }


def batch_samples(samples, batch_size):
    batched_samples = []
    for i in range(0, len(samples), batch_size):
        batch_sample = samples[i:i + batch_size]
        batched_samples.append(batch_sample)
    return batched_samples


def load_samples(dirname_root, max_sample = -1):
    samples = []
    for dirname_sub in sorted(os.listdir(dirname_root)):
        dirname = os.path.join(dirname_root, dirname_sub)
        if not os.path.isdir(dirname):
            continue
        print(dirname)
        sample = load_sample(dirname)
        samples.append(sample)
        if max_sample > 0 and len(samples) >= max_sample:
            break
    return samples


def save_batch(batch, dirname_sample):
    if not os.path.exists(dirname_sample):
        os.makedirs(dirname_sample)

    bs = batch['img_ref_tensor'].shape[0]
    
    for i in range(bs):
        dirname_batch = os.path.join(dirname_sample, f'batch_{i}')
        save_item_tensor(
            dirname_sample=dirname_batch,
            img_tgt_tensor = batch["img_tgt_tensor"][i] if "img_tgt_tensor" in batch else None,
            bkgd_tgt_tensor = batch["bkgd_tgt_tensor"][i],
            domain_tgt_tensor = batch["domain_tgt_tensor"][i],
            style_tgt_tensor = batch["style_tgt_tensor"][i],
            img_ref_tensor = batch["img_ref_tensor"][i],
            reid_ref_tensor = batch["reid_ref_tensor"][i],
            pose_ref_tensor = batch["pose_ref_tensor"][i],
            attention_mask = batch["attention_mask"][i],
        )
    print(batch['vis_tgt_tensor'])
    print(batch['vis_ref_tensor'])
    print(batch['text_tgt_list'])
    print(batch['attention_mask'])
    

    
def save_imgs_tensor(dirname, transform, img_tensor, attention_mask=None):
    img_pil_list = []
    for i in range(img_tensor.shape[0]):
        if attention_mask is not None and not attention_mask[i].item():
            break
        img_pil = transform(img_tensor[i])
        img_pil_list.append(img_pil)
    save_imgs(img_pil_list, dirname)
        
    
# just for check
def save_item_tensor(
    dirname_sample,
    img_tgt_tensor=None, bkgd_tgt_tensor=None,
    domain_tgt_tensor=None, style_tgt_tensor=None, 
    img_ref_tensor=None, reid_ref_tensor=None, pose_ref_tensor=None, attention_mask=None
):
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
    
    if not os.path.exists(dirname_sample):
        os.makedirs(dirname_sample)
    
    if img_tgt_tensor is not None:
        dirname_tgt = os.path.join(dirname_sample, "img_tgt")
        save_imgs_tensor(dirname_tgt, to_pil, img_tgt_tensor)
    if bkgd_tgt_tensor is not None:
        dirname_tgt = os.path.join(dirname_sample, "bkgd_tgt")
        save_imgs_tensor(dirname_tgt, to_pil, bkgd_tgt_tensor)
    if domain_tgt_tensor is not None:
        dirname_ref = os.path.join(dirname_sample, "domain_tgt")
        save_imgs_tensor(dirname_ref, to_pil_imgnet, domain_tgt_tensor)
    if style_tgt_tensor is not None:
        dirname_ref = os.path.join(dirname_sample, "style_tgt")
        save_imgs_tensor(dirname_ref, to_pil_imgnet, style_tgt_tensor)
    if img_ref_tensor is not None:
        dirname_tgt = os.path.join(dirname_sample, "img_ref")
        save_imgs_tensor(dirname_tgt, to_pil, img_ref_tensor, attention_mask)
    if reid_ref_tensor is not None:
        dirname_tgt = os.path.join(dirname_sample, "reid_ref")
        save_imgs_tensor(dirname_tgt, to_pil, reid_ref_tensor, attention_mask)
    if pose_ref_tensor is not None:
        dirname_tgt = os.path.join(dirname_sample, "pose_ref")
        save_imgs_tensor(dirname_tgt, to_pil, pose_ref_tensor, attention_mask)