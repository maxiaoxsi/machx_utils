from ntpath import dirname
import os
import json
import select
from unicodedata import category
from tqdm import tqdm
from PIL import Image
import numpy as np
import random


class Json:
    def __init__(self, dirname, subdataset) -> None:
        self._dirname = dirname
        self._subdataset = subdataset
        self.load_json()


    def load_json(self):
        jsonpath = self.get_dirname("annot") + ".json"
        if not os.path.exists(jsonpath):
            self._json = None
            print("json file not exists!")
            return
        with open(jsonpath, 'r', encoding='utf-8') as f:
            self._json = json.load(f)

    
    def save_json(self):
        jsonpath = self.get_dirname("annot") + ".json"
        version = self._json['info']['version']
        with open(f"{jsonpath}_{version}", 'w', encoding='utf-8') as f:
            print(f"file saved in {jsonpath}_{version}")
            json.dump(self._json, f, indent=4)


    def len_categories(self, key_parent):
        return sum(1 for item in self._json["categories"] if item["supercategory"] == key_parent)


    def get_categories(self, keys_parent):
        return [item["personid"] for item in self._json["categories"] if item["supercategory"] in keys_parent]


    def get_filename(self, id, is_img = False, is_annot = False, type = "reid"):
        if is_img:
            return self._json["images"][id]["filename"]   
        elif is_annot:
            if type == "reid":
                id = self.get_imgid(id, is_annot=True)
                return self.get_filename(id, is_img=True)
            elif type in ["skeleton", "render", "clipreid", "smplx"]:
                return self._json["annotations"][id][type]


    def get_dirname(self, tgtdir):
        if tgtdir == "annot":
            dirname_tgt = "annot"
        elif tgtdir not in self._json["info"]:
            dirname_tgt = tgtdir
        else:
            dirname_tgt = self._json["info"][tgtdir]
        if not dirname_tgt.startswith('/'):
            dirname_tgt = os.path.join(self._dirname, dirname_tgt)
        dirname_tgt = os.path.join(dirname_tgt, self._subdataset)
        return dirname_tgt


    def get_path(self, tgtdir, filename, is_img = False, is_annot = False, type = "reid", suf = None):
        if is_img or is_annot:
            filename = self.get_filename(filename, is_img=is_img, is_annot=is_annot, type=type)
        if suf is not None:
            filename = filename.replace(filename.split('.')[-1], suf)
        dirname_tgt = self.get_dirname(tgtdir=tgtdir)
        path = os.path.join(dirname_tgt, filename)
        return path


    def get_imgid(self, id, is_annot = False):
        if is_annot:
            return self._json["annotations"][id]["imgid"]


    def get_personid(self, id, is_annot = False, is_img = False):
        if is_annot:
            imgid = self.get_imgid(id, is_annot=True)
            personid = self._json["images"][imgid]["personid"]
            return personid
        if is_img:
            personid = self._json["images"][id]["personid"]
            return personid
        return None


    def get_categoryid(self, id, is_category = False, is_annot = False):
        if is_category:
            return self._json["categories"][id]["id"]
        if is_annot:
            return self._json["annotations"][id]["categoryid"]


    def get_supercategoryid(self, id, is_annot = False, is_category = False):
        if is_annot:
            id = self.get_categoryid(id, is_annot=True)   
        while self._json["categories"][id]["supercategory"] != "-1":
            id = self._json["categories"][id]["supercategory"]
        return id


    def get_image_info(self, path_image):
        """获取图片的基本信息"""
        try:
            with Image.open(path_image) as img:
                width, height = img.size
                return width, height
        except Exception as e:
            print(f"无法读取图片 {path_image}: {e}")
            return None, None



class RealPersonJson(Json):
    def __init__(self, dirname, subdataset) -> None:
        super().__init__(dirname, subdataset)


    def load_json(self):
        super().load_json()
        if not self._json:
            return

        self._categories = {}        
        
        for category in self._json['categories']:
            personid = category['personid']
            self._categories[personid] = category

        
    def __contains__(self, item):
        return item in self._categories


    def get_visible(self, id, is_img = False):
        if is_img:
            return self._json['images'][id]['visible']


    def get_fea_clipreid(self, annot):
        if isinstance(annot, int):
            annot = self._json["annotations"][annot]
        filename_clipreid = annot["clipreid"]
        if filename_clipreid == "-1":
            return None
        path_clipreid = self.get_path("clipreid", filename_clipreid)
        if not os.path.exists(path_clipreid):
            return None
        fea_clipreid = np.load(path_clipreid)
        return fea_clipreid


    def get_score_clipreid(self, fea1_clipreid, fea2_clipreid):
        # 确保向量是一维的
        if fea1_clipreid is None or fea2_clipreid is None:
            return -1
        fea1_clipreid = fea1_clipreid.flatten()
        fea2_clipreid = fea2_clipreid.flatten()
        # 计算余弦相似度
        return np.dot(fea1_clipreid, fea2_clipreid) / (np.linalg.norm(fea1_clipreid) * np.linalg.norm(fea2_clipreid))


    def get_pose_tgt(self, annotid):
        render_tgt = self.get_path("render", annotid, is_annot=True, type="render")
        skeleton_tgt = self.get_path("skeleton", annotid, is_annot=True, type="skeleton")
        if not os.path.exists(render_tgt):
            return skeleton_tgt, None
        else:
            if random.random() < 0.5:
                return render_tgt, render_tgt
            else:
                return skeleton_tgt, render_tgt


    def get_images(self, personid):
        '''
        * para personid:int or str 
        * para imgid: int idx or img in categories['images']
        * ans images: list: [imgid]
        '''
        if isinstance(personid, int):
            images = self._json["categories"][personid]["images"]
        elif isinstance(personid, str):
            images = self._categories[personid]["images"]
        return images


    def get_img_tgt(
        self, 
        personid,
        imageid,
    ):
        '''
         * para personid: int or str
         * para imgid: int 
         * ans img_tgt: str, path
         * ans pose_tgt: str, path
         * ans render_tgt: str, path
        '''
        images = self.get_images(personid)
        if imageid <= -1:
            imgid = random.choice(images)
        if imageid < len(images):
            imgid = images[imageid % len(images)] 
        img_tgt = self.get_path("reid", imgid, is_img=True)
        pose_tgt, render_tgt = self.get_pose_tgt(imgid)
        vis_tgt = self.get_visible(imgid, is_img=True)
        vis_tgt = f'a {vis_tgt} photo of a people.'
        return img_tgt, pose_tgt, render_tgt, vis_tgt

    def get_img_ref(self, imgid):
        img_ref = self.get_path("reid", imgid, is_img=True)
        pose_ref = self.get_pose_tgt(imgid)[0]
        return img_ref, pose_ref


    def get_imgs_ref(self, personid, mode="shuffle", max_img = 5):
        if mode == "shuffle":
            gallery_sorted = self._json["annotations"][annotid]["gallery_sorted"]
            num_to_select = random.randint(1, min(max_img, len(gallery_sorted)))
            selected_images = random.sample(gallery_sorted, num_to_select)
            imgs_ref = [self.get_path("reid", selected_annotid, is_annot=True) for selected_annotid in selected_images]
            poses_ref = [self.get_pose_tgt(selected_annotid)[0] for selected_annotid in selected_images]
            return imgs_ref, poses_ref


    def get_item(
        self,
        personid, 
        frameid, 
        imgid,
    ):
        person = self._person[personid]
        if imgid == -1:
            annotid = random.choice(list(person["images"].values()))[0]
        else:
            if imgid not in person["images"]:
                print("imgid key not found!")
                exit()
            annotid = person["images"][imgid][0]
        img_tgt = self.get_path("reid", annotid, is_annot=True)
        pose_tgt, render_tgt = self.get_pose_tgt(annotid)
        imgs_ref, poses_ref = self.get_imgs_ref(annotid)
        return img_tgt, pose_tgt, render_tgt, imgs_ref, poses_ref
       

    def check_ext(self, filename, is_img = False):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        if is_img:
            return any(filename.lower().endswith(ext) for ext in image_extensions)


    def process_batch(self, process_method, tgtdir, suf, batch_size):
        data_batch = {
            "image_list":[],
            "tgt_list":[],
        }
        for image in tqdm(self._json["images"]):
            path_reid = self.get_path("image", image["filename"])
            path_clipreid = self.get_path(tgtdir, image["filename"], suf=suf)
            personid = self.get_personid(image["id"], is_img=True)
            if not personid.isnumeric() or int(personid) <= 0:
                continue
            if os.path.exists(path_clipreid):
                continue
            data_batch["image_list"].append(path_reid)
            data_batch["tgt_list"].append(path_clipreid)
            if len(data_batch["image_list"]) == batch_size:
                process_method(data_batch)
                data_batch["image_list"] = []
                data_batch["tgt_list"] = []
        if data_batch["image_list"] is not []:
            process_method(data_batch)



class RealPersonJsonInitializer(RealPersonJson):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025,
        path_reid="image",
        path_skeleton="skeleton",
        path_render="render",
        path_smplx="smplx",
        path_clipreid="clipreid"
    ) -> None:
        self._dirname = dirname
        self._subdataset = subdataset
        self._init_json(year, path_reid, path_skeleton, path_render, path_smplx, path_clipreid)


    def _init_json(self, year, path_reid, path_skeleton, path_render, path_smplx, path_clipreid):
        self._json = {
            "info": {
                "year": year,
                "version": "1.0",
                "subdataset": f"{self._subdataset}",
                "description": f"RealPerson {self._dirname} {self._subdataset}.",
                "reid": path_reid,
                "skeleton": path_skeleton,
                "render": path_render,
                "smplx": path_smplx,
                "clipreid": path_clipreid
            },
            "licenses": [],
            "images": [],
            "categories": [],
            "licenses": [],
            "annotations": []
        }
        self._init_images()
        self._init_annot()
        self._init_categories()
        self.save_json()


    def _init_images(self):
        dirname = self.get_dirname("reid")
        images = self.traverse_images(dirname)
        images.sort(key=lambda x: (int(x['personid']), int(x['camid']), x["filename"]))
        images_new = []
        for i, image in enumerate(images):
            image_new = {
                "id": i,
                "filename": image["filename"],
                "width": image["width"],
                "height": image["height"],
                "visible": image["visible"],
                "personid": image["personid"],
                "camid": image["camid"],
            }
            images_new.append(image_new)
        self._json["images"] = images_new

    
    def _init_annot(self):
        id = 0
        for image in self._json['images']:
            path_skeleton = self.get_path("skeleton", image['filename'])
            path_render = self.get_path("render", image['filename'])
            path_smplx = self.get_path("smplx", image['filename'])
            path_smplx = path_smplx.replace(path_smplx.split('.')[-1], 'npz')
            path_clipreid=self.get_path("clipreid", image['filename'])
            path_clipreid = path_clipreid.replace(path_clipreid.split('.')[-1], 'npy')
            item = {}
            item["id"] = id
            item["imgid"] = image["id"]
            if not os.path.exists(path_skeleton):
                item["skeleton"] = "-1"
            else:
                dir_skeleton = self.get_dirname("skeleton")
                item["skeleton"] = path_skeleton[len(dir_skeleton) + 1:]
            
            if not os.path.exists(path_render):
                item["render"] = "-1"
            else:
                dir_render = self.get_dirname("render")
                item["render"] = path_render[len(dir_render) + 1:]
            
            if not os.path.exists(path_clipreid):
                item["clipreid"] = "-1"
            else:
                dir_clipreid = self.get_dirname("clipreid")
                item["clipreid"] = path_clipreid[len(dir_clipreid) + 1:]
            
            if not os.path.exists(path_smplx):
                item["smplx"] = "-1"
            else:
                dir_smplx = self.get_dirname("smplx")
                item["smplx"] = path_smplx[len(dir_smplx) + 1:]
                from machx_utils.smplx import SmplxPara
                smplxpara = SmplxPara(smplxpara=path_smplx)
                drn, _, mark_drn = smplxpara.init_drn()
                item["drn"] = drn 
                item["mark_drn"] = mark_drn
            id = id + 1
            self._json["annotations"].append(item)
        

    def _init_category_item(self, id, personid, supercategotyid):
        item = {
            "id": id,
            "personid": personid,
            "supercategory": supercategotyid,
            "images": [],
            "front": [],
            "back": [],
            "left": [],
            "right": [],
        }
        return item


    def get_categoryid(self, key, is_personid = False):
        if is_personid:
            for category in self._json["categories"]:
                if category["personid"] == key:
                    return category["id"]
    

    def get_drn(self, key, is_annot = False):
        if is_annot:
            if "drn" not in self._json["annotations"][key]:
                return None
            return self._json["annotations"][key]["drn"]
        

    def _add_img_to_category(self, categotyid, imgid):
        category = self._json["categories"][categotyid]
        if imgid in category["images"]:
            return
        category["images"].append(imgid)
        self._json["annotations"][imgid]["categoryid"] = categotyid
        drn = self.get_drn(imgid, is_annot=True)
        if drn is not None:
            category[drn].append(imgid)
    

    def _init_categories(self):
        id = 0
        personids = []
        for image in self._json['images']:
            personid = image['personid']
            imgid = image['id']
            if personid not in personids:
                personids.append(personid)
                categoryid = id
                category = self._init_category_item(categoryid, personid, "-1")
                self._json["categories"].append(category)
                id = id + 1
            else:
                categoryid = self.get_categoryid(personid, is_personid=True)
            self._add_img_to_category(categoryid, imgid)
        self._init_reference()
        

    def _init_reference(self):
        for annot in self._json['annotations']:
            categoryid = annot["categoryid"]
            imgs = self._json['categories'][categoryid]["images"]
            fea_query_clipreid = self.get_fea_clipreid(annot)
            annots_with_scores = []
            for annotid in imgs:
                annot_gallery = self._json["annotations"][annotid]
                fea_gallery_clipreid = self.get_fea_clipreid(annot_gallery)
                score = self.get_score_clipreid(fea_query_clipreid, fea_gallery_clipreid)
                score = float(score)
                annots_with_scores.append((annotid, score))

            sorted_annots_with_scores = sorted(annots_with_scores, 
                key=lambda x: x[1], 
                reverse=True
            )
            
            sorted_annots = [item[0] for item in sorted_annots_with_scores]
            annot["reference"] = sorted_annots


    def traverse_images(self, dirname):
        images = []
         
        if not os.path.exists(dirname):
            print(f"{dirname} not exists, json file not created!")
            exit()

        for root, dirs, files in os.walk(dirname):
            for file in files:
                if self.check_ext(file, is_img=True):
                    file_path = os.path.join(root, file)
                    width, height, id_person, id_camera, visible = self.get_image_info(file_path, file)
                    if width and height and id_person and id_camera:
                        filepath = os.path.join(root, file)
                        filesubpath = filepath[len(dirname)+1:]
                        images.append({
                            "filename": filesubpath,
                            "height": height,
                            "width": width,
                            "visible": visible,
                            "personid": id_person,
                            "camid": id_camera,
                        })
        return images

    

class MarketInitializer(RealPersonJsonInitializer):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025, 
        path_reid="image", 
        path_skeleton="skeleton", 
        path_render="render", 
        path_smplx="smplx", 
        path_clipreid="clipreid"
    ) -> None:
        super().__init__(dirname, subdataset, year, path_reid, path_skeleton, 
                path_render, path_smplx, path_clipreid)
    

    def get_image_info(self, path_image, filename):
        """获取图片的基本信息"""
        width, height = super().get_image_info(path_image)
        if width is None:
            return None, None, None, None, None
        id_person = filename.split('_')[0]
        id_camera = filename.split('_')[1].split('c')[1].split('s')[0]
        if not id_person.isdigit() or int(id_person) < 1:
            return None, None, None, None, None  
        return width, height, id_person, id_camera, "visible"



class SYSUMM01Initializer(RealPersonJsonInitializer):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025, 
        path_reid="image", 
        path_skeleton="skeleton", 
        path_render="render", 
        path_smplx="smplx", 
        path_clipreid="clipreid"
    ) -> None:
        super().__init__(dirname, subdataset, year, path_reid, path_skeleton, 
                path_render, path_smplx, path_clipreid)
    
    def get_image_info(self, path_image, filename):
        """获取图片的基本信息"""
        width, height = super().get_image_info(path_image)
        if width is None:
            return None, None, None, None, None
        id_person = path_image.split('/')[-2]
        id_camera = path_image.split('/')[-3][3]
        if not id_person.isdigit() or int(id_person) < 1:
            return None, None, None, None, None  
        if id_camera in ['3', '6']:
            visible = "infrared"
        else:
            visible = "visible"
        return width, height, id_person, id_camera, visible



class MSMT17V1Initializer(RealPersonJsonInitializer):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025, 
        path_reid="image", 
        path_skeleton="skeleton", 
        path_render="render", 
        path_smplx="smplx", 
        path_clipreid="clipreid"
    ) -> None:
        super().__init__(dirname, subdataset, year, path_reid, path_skeleton, 
                path_render, path_smplx, path_clipreid)
    

    def get_image_info(self, path_image, filename):
        """获取图片的基本信息"""
        width, height = super().get_image_info(path_image)
        if width is None:
            return None, None, None, None, None
        id_person = filename.split('_')[0]
        id_camera = filename.split('_')[1].split('c')[1]
        if not id_person.isdigit() or int(id_person) < 1:
            return None, None, None, None, None  
        return width, height, id_person, id_camera, "visible"



class OCCReIDInitializer(RealPersonJsonInitializer):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025, 
        path_reid="image", 
        path_skeleton="skeleton", 
        path_render="render", 
        path_smplx="smplx", 
        path_clipreid="clipreid"
    ) -> None:
        super().__init__(dirname, subdataset, year, path_reid, path_skeleton, 
                path_render, path_smplx, path_clipreid)
    

    def get_image_info(self, path_image, filename):
        """获取图片的基本信息"""
        width, height = super().get_image_info(path_image)
        if width is None:
            return None, None, None, None, None
        id_person = filename.split('_')[0]
        id_camera = "1"
        if not id_person.isdigit() or int(id_person) < 1:
            return None, None, None, None, None  
        return width, height, id_person, id_camera, "visible"



class DUKEInitializer(RealPersonJsonInitializer):
    def __init__(
        self, 
        dirname, 
        subdataset, 
        year=2025, 
        path_reid="image", 
        path_skeleton="skeleton", 
        path_render="render", 
        path_smplx="smplx", 
        path_clipreid="clipreid"
    ) -> None:
        super().__init__(dirname, subdataset, year, path_reid, path_skeleton, 
                path_render, path_smplx, path_clipreid)
    

    def get_image_info(self, path_image, filename):
        """获取图片的基本信息"""
        width, height = super().get_image_info(path_image)
        if width is None:
            return None, None, None, None, None
        id_person = filename.split('_')[0]
        id_camera = filename.split('_')[1].split('c')[1]
        if not id_person.isdigit() or int(id_person) < 1:
            return None, None, None, None, None
        return width, height, id_person, id_camera, "visible"


