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


class Dataset:
    def __init__(self, dirname, datasetname, subdataset) -> None:
        self._init_transforms()
        self._json = Json(dirname, datasetname, subdataset)

    def _init_transforms(self):
        self._transforms = {}
        self._transforms["reid"]=transforms.Compose(
            [
                Scale2D(128, 256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5]),
            ]
        )

    def get_image_tensor(self, type_transforms, path_image):
        image_pil = Image.open(path_image)
        return self._transforms[type_transforms](image_pil)