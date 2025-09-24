import torchvision.transforms as transforms
from PIL import Image
import random
from machx_utils.realperson import Json

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

class TransformsSet:
    def __init__(self) -> None:
        self._transforms = {}
        self._transforms["reid"]=transforms.Compose(
            [
                Scale2D(128, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5]),
            ]
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._transforms[args[0]]

class Dataset:
    def __init__(
        self, 
        dirname, 
        datasetname, 
        subdataset,
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
        self._transforms_set = TransformsSet()
        self._json = Json(dirname, datasetname, subdataset)
        self._img_size=img_size

    def __len__(self):
        if not self._len:
            self._len = self._json.len_categories("-1") 
        return self._len

    def __contains__(self, key):
        if not self._keys:
            self._keys = self._json.get_categories("-1")
        return key in self._keys

    def __getitem__(self, idx):
        return self.get_item(
            id_person=idx,
            idx_vid=-1,
            idx_img=-1,
        )

    def get_item(self, id_person, id_frame, id_img):
        sample = self._json.get_item(
            id_person, 
            id_frame, 
            id_img,
            is_select_bernl = self._is_select_bernl,
            is_select_repeat = self._is_select_repeat,
            rate_mask_aug = self.rate_mask_aug,
        )

        img_ref, img_reid = self.get_img_ref(sample['imglist_ref'])
        img_tgt = self.get_img_tgt(sample['img_tgt'])
        img_bkgd = self.get_img_bkgd(sample['img_bkgd'])
        pose_skeleton, pose_render, pose_rgb = self.get_pose(
            sample['img_skeleton'],
            sample['img_render'],
            sample['img_tgt'],
        )
        return {
            "img_ref": img_ref,
            "img_tgt": img_tgt,
            "img_reid": img_reid,
            "img_bkgd": img_bkgd,
            "pose_skeleton": pose_skeleton,
            "pose_render": pose_render,
            "pose_rgb": pose_rgb,
        }

    def get_image_tensor(self, type_transforms, path_image):
        image_pil = Image.open(path_image)
        return self._transforms_set(type_transforms)(image_pil)



    