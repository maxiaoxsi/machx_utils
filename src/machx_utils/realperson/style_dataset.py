from machx_utils.realperson import RealPersonJson
import random
import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
from PIL import Image



import torch
import torch.nn as nn
from einops import rearrange

class PatchShuffler(nn.Module):
    """一个用于随机打乱图像块的模块（单张图像版本）"""
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        输入: image tensor of shape (C, H, W)
        输出: shuffled image tensor of shape (C, H, W)
        """
            
        C, H, W = image.shape
        p = self.patch_size
        
        # 将图像分割成 patches: (C, H, W) -> (Num_Patches, C, Patch_H, Patch_W)
        patches = rearrange(image, 'c (h p1) (w p2) -> (h w) c p1 p2', p1=p, p2=p)
        num_patches = patches.shape[0]

        # 生成随机索引
        rand_indices = torch.randperm(num_patches, device=image.device)
        
        # 根据随机索引打乱 patches
        shuffled_patches = patches[rand_indices]
        
        # 重新拼接成图像
        h_patches = H // p
        w_patches = W // p
        shuffled_image = rearrange(shuffled_patches, '(h w) c p1 p2 -> c (h p1) (w p2)', 
                                  h=h_patches, w=w_patches, p1=p, p2=p)
        return shuffled_image



class StyleDataset:
    def __init__(self, jsons):
        self._jsons = []
        self._len_jsons = []
        for json in jsons:
            self._jsons.append(json)
            self._len_jsons.append(len(json))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_shuffle = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
            PatchShuffler(patch_size=16),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return sum(self._len_jsons)
    
    def __getitem__(self, idx):
        for json, len_json in zip(self._jsons, self._len_jsons):
            if idx < len_json:
                json_selected = json
                datasetid = json_selected.get_datasetid()
                break
            else:
                idx -= len_json
        path_img = json_selected.get_path("reid", idx, is_img=True)
        domainid = self.get_domainid(datasetid)
        camid = int(json_selected.get_camid(idx, is_img=True))
        img_pil = Image.open(path_img).convert("RGB")
        img_ma_tensor = self.transform(img_pil)
        img_mi_tensor = self.transform_shuffle(img_pil) 
        img_mi_pos_tensor = self.transform_shuffle(img_pil) 
        return {
            "img_ma_tensor": img_ma_tensor, 
            "img_mi_tensor": img_mi_tensor, 
            "img_mi_pos_tensor":img_mi_pos_tensor, 
            "domainid":domainid, 
            "camid":camid
        }

    def get_domainid(self, datasetid):
        if "market" in datasetid.lower():
            return 0
        elif "sysumm" in datasetid.lower():
            return 1
        elif "msmt" in datasetid.lower():
            return 2
        elif "duke" in datasetid.lower():
            return 3
        elif "occreid" in datasetid.lower():
            return 4
