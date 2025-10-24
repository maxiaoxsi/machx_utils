# from turtle import width
import torchvision.transforms as transforms
from PIL import Image
import random
from machx_utils.realperson import RealPersonJson
import torchvision.transforms.functional as F
import numpy as np
import torch
import os


class Scale2D:
    def __init__(self, width, height, interpolation=Image.BILINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation
    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class Scale1D:
    def __init__(self, size_tgt, interpolation=Image.BILINEAR):
        self._size_tgt = size_tgt
        self._interpolation = interpolation
    
    def __call__(self, img):
        w, h = img.size
        if w > h:
            width_tgt = self._size_tgt
            height_tgt = int(self._size_tgt / w * h)
        else:
            width_tgt = int(self._size_tgt / h * w)
            height_tgt = self._size_tgt
        return img.resize((width_tgt, height_tgt), self._interpolation)


class RandomCrop:
    def __init__(self, width_scale, height_scale):
        self._width_scale = width_scale
        self._height_scale = height_scale
        self._random_w = np.random.uniform(width_scale[0], width_scale[1])
        self._random_h = np.random.uniform(height_scale[0], height_scale[1])
        
    
    def _getPoint(self, scale, w):
        w_target = int(w * scale)
        w_target = max(1, w_target)
        w_target = min(w_target, w)
        w_start = random.randint(0, w - w_target)
        w_end = w_start + w_target
        return w_start, w_end
    
    def __call__(self, img):
        w, h = img.size
        w_start, w_end = self._getPoint(self._random_w, w)
        h_start, h_end = self._getPoint(self._random_h, h)
        return img.crop((w_start, h_start, w_end, h_end))


class PadToBottomRight:
    def __init__(self, target_size, fill=0):
        self.target_size = target_size  # 目标尺寸 (W, H)
        self.fill = fill  # 填充值

    def __call__(self, img):
        """
        img: Tensor of shape [C, H, W]
        Returns: Padded Tensor of shape [C, target_H, target_W]
        """
        _, h, w = img.shape
        pad_w = max(self.target_size[0] - w, 0)  # 右侧需填充的宽度
        pad_h = max(self.target_size[1] - h, 0)  # 底部需填充的高度
        padding = (0, 0, pad_w, pad_h)  
        img_padded = F.pad(img, padding, fill=self.fill)
        return img_padded


class TransformsSet:
    def __init__(self, img_size, width_scale, height_scale, rate_random_erase) -> None:
        random_crop = RandomCrop(width_scale, height_scale)
        self._transforms = {}

        self._transforms["reid"]=transforms.Compose(
            [
                Scale2D(128, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5]),
            ]
        )

        self._transforms["ref"] = transforms.Compose(
            [
                random_crop,
                Scale1D(img_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.RandomErasing(p=rate_random_erase, scale=(0.12, 0.37), ratio=(0.3, 3.3), value=0, inplace=False),
                PadToBottomRight(target_size=img_size, fill=0),
            ]
        ) 
        
        self._transforms["norm"] = transforms.Compose(
            [
                random_crop,
                Scale1D(img_size[0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                PadToBottomRight(target_size=img_size, fill=0),
            ]
        )

    def __call__(self, type):
        return self._transforms[type]

class Dataset:
    def __init__(
        self, 
        jsons_tgt,
        jsons_ref,
        is_select_bernl=True,
        is_select_repeat=True,
        rate_random_erase=0.5,
        rate_dropout_back=0.2,
        img_size=(512, 512),
        width_scale=(1, 1),
        height_scale=(1, 1),
        n_frame=10,
    ) -> None:
        self._jsons_tgt = jsons_tgt
        self._jsons_ref = jsons_ref
        self._img_size = img_size
        self._width_scale = width_scale
        self._height_scale = height_scale
        self._rate_random_erase = rate_random_erase
        self._rate_dropout_back = rate_dropout_back
        self._init_personid_list()

    def _init_personid_list(self):
        self._personid_list = []
        for json_tgt in self._jsons_tgt:
            personid_list = json_tgt.get_categories(["-1", -1])
            for personid in personid_list:
                if personid in self._personid_list:
                    continue
                for json_ref in self._jsons_ref:
                    if personid in json_ref:
                        self._personid_list.append(personid)
                        break
        # print(self._personid_list)
        

    def __len__(self):
        return len(self._personid_list)


    def __contains__(self, item):
        return item in self._personid_list


    def __getitem__(self, idx):
        personid = self._personid_list[idx]
        img_tgt_list, pose_tgt_list, render_tgt_list, vis_tgt, img_ref_list, pose_ref_list, vis_ref_list = self.get_item(
            personid=personid,
            imgid=-1,
        )
        img_tgt_tensor, bkgd_tgt_tensor, pose_tgt_tensor = self.get_img_tgt(
            img_tgt_list, pose_tgt_list, render_tgt_list
        )
        img_ref_tensor, reid_ref_tensor, pose_ref_tensor = self.get_imgs_ref(img_ref_list, pose_ref_list)
        return {
            "img_tgt_tensor": img_tgt_tensor,
            "bkgd_tgt_tensor": bkgd_tgt_tensor,
            "pose_tgt_tensor": pose_tgt_tensor,
            "vis_tgt": vis_tgt,
            "img_ref_tensor": img_ref_tensor,
            "reid_ref_tensor": reid_ref_tensor,
            "pose_ref_tensor": pose_ref_tensor,
            "vis_ref_list": vis_ref_list,
        }


    def get_item_tgt(self, personid, imgid):
        jsons_tgt = [json for json in self._jsons_tgt if personid in json]
        json_tgt = random.choice(jsons_tgt)
        img_tgt_list, pose_tgt_list, render_tgt_list, vis_tgt = json_tgt.get_img_tgt(personid, imgid)
        return img_tgt_list, pose_tgt_list, render_tgt_list, vis_tgt


    def random_select_simple(images, n):
        all_elements = [(i, imgid) for i, subimages in enumerate(images) for imgid in subimages]
        return random.sample(all_elements, min(n, len(all_elements)))


    def get_item_ref(self, personid, n_max):
        jsons_ref = [json for json in self._jsons_ref if personid in json]
        images = [json.get_images(personid) for json in jsons_ref]
        images = [(i, imgid) for i, subimages in enumerate(images) for imgid in subimages]
        n_max = random.randint(1, n_max)
        images_selected = random.sample(images, min(n_max, len(images)))
        img_ref_list = []
        pose_ref_list = []
        vis_ref_list = []
        for (i, imgid) in images_selected:
            img_ref, pose_ref = jsons_ref[i].get_img_ref(imgid)
            vis_ref = jsons_ref[i].get_visible(imgid)
            img_ref_list.append(img_ref)
            pose_ref_list.append(pose_ref)
            vis_ref_list.append(vis_ref)
        return img_ref_list, pose_ref_list, vis_ref_list


    def get_item(self, personid, imgid):
        img_tgt_list, pose_tgt_list, render_tgt_list, vis_tgt = self.get_item_tgt(personid, imgid)
        img_ref_list, pose_ref_list, vis_ref_list = self.get_item_ref(personid, 8)
        return img_tgt_list, pose_tgt_list, render_tgt_list, vis_tgt, img_ref_list, pose_ref_list, vis_ref_list


    def get_img_tgt(self, img_tgt_list, pose_tgt_list, render_tgt_list):
        transforms_set = TransformsSet(self._img_size, 
            self._width_scale, self._height_scale, self._rate_random_erase)
        from machx_utils.realperson import make_mask
        img_tgt_tensor_list = []
        bkgd_tgt_tensor_list = []
        pose_tgt_tensor_list = []
        for (img_tgt, pose_tgt, render_tgt) in zip(
            img_tgt_list, pose_tgt_list, render_tgt_list
        ):
            _, _, bkgd_tgt = make_mask(img_tgt, render_tgt)
            if random.random() < self._rate_dropout_back:
                bkgd_tgt = Image.new('RGB', bkgd_tgt.size, (0, 0, 0))
            img_tgt_tensor = self.get_image_tensor(transforms_set, "norm", img_tgt)
            pose_tgt_tensor = self.get_image_tensor(transforms_set, "norm", pose_tgt)
            bkgd_tgt_tensor = self.get_image_tensor(transforms_set, "norm", bkgd_tgt)
            img_tgt_tensor_list.append(img_tgt_tensor)
            bkgd_tgt_tensor_list.append(bkgd_tgt_tensor)
            pose_tgt_tensor_list.append(pose_tgt_tensor)
        img_tgt_tensor = torch.stack(img_tgt_tensor_list, dim=0)
        bkgd_tgt_tensor = torch.stack(bkgd_tgt_tensor_list, dim=0)
        pose_tgt_tensor = torch.stack(pose_tgt_tensor_list, dim=0)
        return img_tgt_tensor, bkgd_tgt_tensor, pose_tgt_tensor # (f c h w)


    def get_imgs_ref(self, img_ref_list, pose_ref_list):
        transforms_set = TransformsSet(self._img_size, 
            self._width_scale, self._height_scale, self._rate_random_erase)
        img_ref_tensor_list = []
        reid_ref_tensor_list = []
        pose_ref_tensor_list = []
        for (img_ref, pose_ref) in zip(img_ref_list, pose_ref_list):
            transforms_set = TransformsSet(self._img_size, 
                self._width_scale, self._height_scale, self._rate_random_erase)
            img_ref_tensor = self.get_image_tensor(transforms_set, "ref", img_ref)
            reid_ref_tensor = self.get_image_tensor(transforms_set, "reid", img_ref)
            pose_ref_tensor = self.get_image_tensor(transforms_set, "norm", pose_ref)
            img_ref_tensor_list.append(img_ref_tensor)
            reid_ref_tensor_list.append(reid_ref_tensor)
            pose_ref_tensor_list.append(pose_ref_tensor)
        img_ref_tensor = torch.stack(img_ref_tensor_list, dim=0)
        reid_ref_tensor = torch.stack(reid_ref_tensor_list, dim=0)
        pose_ref_tensor = torch.stack(pose_ref_tensor_list, dim = 0)
        return img_ref_tensor, reid_ref_tensor, pose_ref_tensor

    
    def get_image_tensor(self, transforms_set, type_tansform, image):
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image)
            else:
                image = Image.new('RGB', (512, 512), (0, 0, 0))
        return transforms_set(type_tansform)(image)



    