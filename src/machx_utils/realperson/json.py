from ntpath import dirname
import os
import json
from unicodedata import category
from tqdm import tqdm
from PIL import Image
import numpy as np

def get_image_info(image_path, filename):
    """获取图片的基本信息"""
    if 'market' in image_path.lower():
        id_person = filename.split('_')[0]
        id_camera = filename.split('_')[1].split('c')[1].split('s')[0]
    if not id_person.isdigit() or int(id_person) < 1:
        return None, None, None, None
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height, id_person, id_camera
    except Exception as e:
        print(f"无法读取图片 {image_path}: {e}")
        return None, None, None, None


class Json:
    def __init__(self, dirname, subdataset) -> None:
        self._dirname = dirname
        self._subdataset = subdataset
        self.load_json()


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


    def get_path(self, tgtdir, filename):
        dirname_tgt = self.get_dirname(tgtdir=tgtdir)
        path = os.path.join(dirname_tgt, filename)
        return path


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


    def get_categories(self, key_parent):
        return [item["name"] for item in self._json["categories"] if item["supercategory"] == key_parent]


    def get_supercategory(self, id, is_annot = False, is_category = False):
        if is_annot:
            id = self._json["annotations"][id]["category_id"]
        while self._json["categories"][id]["supercategory"] != "-1":
            id = self._json["categories"][id]["supercategory"]
        return id
        

    def get_category(self, id, is_category = False, is_annot = False):
        if is_category:
            return self._json["categories"][id]



class RealPersonJson(Json):
    def __init__(self, dirname, subdataset) -> None:
        super().__init__(dirname, subdataset)
        self.load_json()


    def load_json(self):
        super().load_json()
        if not self._json:
            return
        # print(self.get_categories("-1"))
        # print(self.len_categories("-1"))
        # print(len(self.get_categories("-1")))
        self._person = {}
        for item in self._json['images']:
            personid = item['personid']
            imgid = item['id']
            if personid not in self._person:
                self._person[personid] = {"imgid": [imgid]}
            else:
                self._person[personid]["imgid"].append(imgid)
        
        for item in self._json['annotations']:
            skeleton = item["skeleton"]
            render = item["render"]
            imgid = item["id"]
            annotid = item["id"]
            if render != "-1" or skeleton != "-1":
                personid = self.get_image(imgid)["personid"]
                self._person[personid][imgid].append(annotid)
        
                
    def get_image(self, imgid):
        if not self._json:
            return None
        return self._json["images"][imgid]


    def get_annot(self, annotid):
        if not self._json:
            return None
        return self._json["annotations"][annotid]


    def get_clipreid(self, annot):
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
        fea1_clipreid = fea1_clipreid.flatten()
        fea2_clipreid = fea2_clipreid.flatten()
        # 计算余弦相似度
        return np.dot(fea1_clipreid, fea2_clipreid) / (np.linalg.norm(fea1_clipreid) * np.linalg.norm(fea2_clipreid))


    def check_ext(self, filename):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)


    def check_image(self, i):
        if not self._json:
            self.load_json()
        print(self._json["images"][i])


    def get_item(
        self,
        id_person, 
        id_frame, 
        id_img,
        is_select_bernl,
        is_select_repeat,
        rate_mask_aug
    ):
        pass


    def process_batch(self, process_method, tgtdir, batch_size):
        data_batch = {
            "image_list":[],
            "tgt_list":[],
        }
        for image in tqdm(self._json["images"]):
            data_batch["image_list"].append(self.get_path("image", image["filename"], None))
            data_batch["tgt_list"].append(self.get_path(tgtdir, image["filename"], 'npy'))
            if len(data_batch["image_list"]) == batch_size:
                process_method(data_batch)
                data_batch["image_list"] = []
                data_batch["tgt_list"] = []
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
        images= self.traverse_images(dirname)
        images.sort(key=lambda x: (int(x['personid']), int(x['camid']), x["filename"]))
        images_new = []
        for i, image in enumerate(images):
            image_new = {
                "id": i,
                "filename": image["filename"],
                "width": image["width"],
                "height": image["height"],
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
            path_clipreid = path_clipreid.replace(path_smplx.split('.')[-1], 'npz')
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
            if not os.path.exists(path_smplx):
                item["smplx"] = "-1"
            else:
                dir_smplx = self.get_dirname("smplx")
                item["smplx"] = path_smplx[len(dir_smplx) + 1:]

            if not os.path.exists(path_clipreid):
                item["clipreid"] = "-1"
            else:
                dir_clipreid = self.get_dirname("clipreid")
                item["clipreid"] = path_smplx[len(dir_clipreid) + 1:]

            if os.path.exists(path_smplx):
                from machx_utils.smplx import SmplxPara
                smplxpara = SmplxPara(smplxpara=path_smplx)
                drn, _, mark_drn = smplxpara.init_drn()
                item["drn"] = drn 
                item["mark_drn"] = mark_drn
            id = id + 1
            self._json["annotations"].append(item)
        

    def _init_categories(self):
        dict_personid = {}
        id = 0
        for image in self._json['images']:
            personid = image['personid']
            imageid = image['id']
            if personid not in dict_personid:
                dict_personid[personid] = id
                item = {
                    "id": id,
                    "name": personid,
                    "supercategory": "-1",
                }
                self._json["categories"].append(item)
                id = id + 1
        
        for image in self._json['images']:
            personid = image['personid']
            imageid = image['id']
            if "drn" not in self._json['annotations'][imageid]:
                self._json["annotations"][imageid]["category_id"] = dict_personid[personid]
                continue
            drn = self._json['annotations'][imageid]['drn']
            personid_wtdrn = f"{personid}_{drn}"
            supercategory = dict_personid[personid]
            if personid_wtdrn not in dict_personid:
                dict_personid[personid_wtdrn] = id
                item = {
                    "id": id,
                    "name": personid_wtdrn,
                    "supercategory": supercategory
                }
                self._json["categories"].append(item)
                id = id + 1
            self._json["annotations"][imageid]["category_id"] = dict_personid[personid_wtdrn]

        dict_categories = {}

        for annot in self._json['annotations']:
            categoryid = annot["category_id"]
            categoryid = self.get_supercategory(categoryid)
            if categoryid not in dict_categories:
                dict_categories[categoryid] = [annot["id"]]
            else:
                dict_categories[categoryid].append(annot["id"])

        for annot in self._json['annotations']:
            categoryid = annot["category_id"]
            categoryid = self.get_supercategory(categoryid)
            annots = dict_categories[categoryid]
            fea_query_clipreid = self.get_clipreid(annot)
            annots_with_scores = []
            for annot_gallery in annots:
                fea_gallery_clipreid = self.get_clipreid(annot_gallery)
                score = self.get_score_clipreid(fea_query_clipreid, fea_gallery_clipreid)
                annots_with_scores.append((annot_gallery, score))

            sorted_annots_with_scores = sorted(annots_with_scores, 
                key=lambda x: x[1], 
                reverse=True
            )
            sorted_annots = [item[0] for item in sorted_annots_with_scores]
            for annot_gallery in sorted_annots:
                fea_gallery_clipreid = self.get_clipreid(annot_gallery)
                print(fea_query_clipreid)
                print(fea_gallery_clipreid)
                score_clipreid = self.get_score_clipreid(fea_query_clipreid, fea_gallery_clipreid)
                print(score_clipreid)
                exit()
                
            
            

    def traverse_images(self, dirname):
        images = []
         
        if not os.path.exists(dirname):
            print(f"{dirname} not exists, json file not created!")
            exit()

        for root, dirs, files in os.walk(dirname):
            for file in files:
                if self.check_ext(file):
                    file_path = os.path.join(root, file)
                    width, height, id_person, id_camera = get_image_info(file_path, file)
                    if width and height and id_person and id_camera:
                        filepath = os.path.join(root, file)
                        filesubpath = filepath[len(dirname)+1:]
                        images.append({
                            "filename": filesubpath,
                            "height": height,
                            "width": width,
                            "personid": id_person,
                            "camid": id_camera,
                        })
        return images


    

