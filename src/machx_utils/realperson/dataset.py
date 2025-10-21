# from turtle import width
import torchvision.transforms as transforms
from PIL import Image
import random
from machx_utils.realperson import RealPersonJson
import torchvision.transforms.functional as F
import numpy as np
import torch


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
        rate_dropout_ref=0.2,
        rate_dropout_back=0.2,
        rate_dropout_manikin=0,
        rate_dropout_skeleton=0.2,
        rate_dropout_rgbguid=1,
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
        print(self._personid_list)
        

    def __len__(self):
        return len(self._personid_list)


    def __contains__(self, item):
        return item in self._personid_list


    def __getitem__(self, idx):
        img_tgt, pose_tgt, render_tgt, vis_tgt, imgs_ref, poses_ref = self.get_item(
            personid=idx,
            imgid=-1,
        )
        img_tgt, bkgd_tgt, pose_tgt = self.get_img_tgt(img_tgt, pose_tgt, render_tgt)
        imgs_ref, reids_ref, poses_ref = self.get_imgs_ref(imgs_ref, poses_ref)
        return {
            "img_tgt": img_tgt,
            "bkgd_tgt": bkgd_tgt,
            "pose_tgt": pose_tgt,
            "vis_tgt": vis_tgt,
            "imgs_ref": imgs_ref,
            "reids_ref": reids_ref,
            "poses_ref": poses_ref,
        }


    def get_item_tgt(self, personid, imgid):
        json_tgt = random.choice(self._jsons_tgt)
        img_tgt, pose_tgt, render_tgt, vis_tgt = json_tgt.get_img_tgt(personid, imgid)
        return img_tgt, pose_tgt, render_tgt, vis_tgt


    def random_select_simple(images, n):
        all_elements = [(i, imgid) for i, subimages in enumerate(images) for imgid in subimages]
        return random.sample(all_elements, min(n, len(all_elements)))


    def get_item_ref(self, personid, n_max):
        images = [json.get_images(personid) for json in self._jsons_ref]
        images = [(i, imgid) for i, subimages in enumerate(images) for imgid in subimages]
        n_max = random.randint(1, n_max)
        images_selected = random.sample(images, min(n_max, len(images)))
        imgs_ref = []
        poses_ref = []
        for (i, imgid) in images_selected:
            img_ref, pose_ref = self._jsons_ref[i].get_img_ref(imgid)
            imgs_ref.append(img_ref)
            poses_ref.append(pose_ref)
        return imgs_ref, poses_ref


    def get_item(self, personid, imgid):
        img_tgt, pose_tgt, render_tgt, vis_tgt = self.get_item_tgt(personid, imgid)
        imgs_ref, poses_ref = self.get_item_ref(personid, 8)
        return img_tgt, pose_tgt, render_tgt, vis_tgt, imgs_ref, poses_ref


    def get_img_tgt(self, img_tgt, pose_tgt, render_tgt):
        transforms_set = TransformsSet(self._img_size, 
            self._width_scale, self._height_scale, self._rate_random_erase)
        from machx_utils.realperson import make_mask
        _, _, bkgd_tgt = make_mask(img_tgt, render_tgt)
        img_tgt = self.get_image_tensor(transforms_set, "norm", img_tgt)
        pose_tgt = self.get_image_tensor(transforms_set, "norm", pose_tgt)
        bkgd_tgt = self.get_image_tensor(transforms_set, "norm", bkgd_tgt)
        return img_tgt, bkgd_tgt, pose_tgt


    def get_imgs_ref(self, imgs_ref, poses_ref):
        transforms_set = TransformsSet(self._img_size, 
            self._width_scale, self._height_scale, self._rate_random_erase)
        imgs_ref_list = []
        reids_ref_list = []
        poses_ref_list = []
        for (img_ref, pose_ref) in zip(imgs_ref, poses_ref):
            transforms_set = TransformsSet(self._img_size, 
                self._width_scale, self._height_scale, self._rate_random_erase)
            reid_ref = self.get_image_tensor(transforms_set, "reid", img_ref)
            img_ref = self.get_image_tensor(transforms_set, "ref", img_ref)
            pose_ref = self.get_image_tensor(transforms_set, "norm", pose_ref)
            imgs_ref_list.append(img_ref)
            reids_ref_list.append(reid_ref)
            poses_ref_list.append(pose_ref)
        imgs_ref = torch.stack(imgs_ref_list, dim=0)
        reids_ref = torch.stack(reids_ref_list, dim=0)
        poses_ref = torch.stack(poses_ref_list, dim = 0)
        return imgs_ref, reids_ref, poses_ref

    
    def get_image_tensor(self, transforms_set, type_tansform, image):
        if isinstance(image, str):
            image = Image.open(image)
        return transforms_set(type_tansform)(image)



    