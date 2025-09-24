import os
import json
from tqdm import tqdm
from PIL import Image

def get_image_info(image_path, filename):
    """获取图片的基本信息"""
    if not 'market' in image_path.lower():
        return
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
        if tgtdir not in self._json["info"]:
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
        if not os.path.exists(self._jsonpath):
            print("json file not exists!")
            return
        with open(self._jsonpath, 'r', encoding='utf-8') as f:
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
        
    


class RealPersonJson(Json):
    def __init__(self, dirname, subdataset) -> None:
        super().__init__(dirname, subdataset)


    def load_json(self):
        super().load_json()
        self._person = {}
        for item in self._json['images']:
            personid = item['personid']
            imageid = item['id']
            if personid not in self._person:
                self._person[personid] = [imageid]
                print(personid)
                print(imageid)
            else:
                self._person[personid].append(imageid)


    def check_ext(self, filename):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)


    def check_image(self, i):
        if not self._json:
            self.load_json()
        print(self._json["images"][i])


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
        path_smplx="smplx"
    ) -> None:
        self._dirname = dirname
        self._subdataset = subdataset
        self._init_json(year, path_reid, path_skeleton, path_render, path_smplx)


    def _init_json(self, year, path_reid, path_skeleton, path_render, path_smplx):
        self._json = {
            "info": {
                "year": year,
                "version": "1.0",
                "subdataset": f"{self._subdataset}",
                "description": f"RealPerson {self._dirname} {self._subdataset}.",
                "reid": path_reid,
                "skeleton": path_skeleton,
                "render": path_render,
                "smplx": path_smplx 
            },
            "licenses": [],
            "images": [],
            "categories": [],
            "licenses": [],
            "annotations": []
        }
        self._init_images()
        self._init_categories()
        self._init_annot()
        self._init_drn()
        self.save_json()


    def _init_images(self):
        path_dataset = self.get_dirname("reid")
        images= self.traverse_images(path_dataset)
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
            item = {}
            item["id"] = id
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
            id = id + 1
            self._json["annotations"].append(item)


    def _init_categories(self):
        list_personid = []
        id = 0
        for image in self._json['images']:
            personid = image['personid']
            if personid not in list_personid:
                list_personid.append(personid)
                item = {
                    "id": id,
                    "name": personid,
                    "supercategory": "-1",
                }
                self._json["categories"].append(item)
                id = id + 1


    def _init_drn(self):
        for item in self._json["images"]:
            itemid = item["id"]
            annot = self._json["annotations"][itemid]
            path_smplx = self.get_path("smplx", annot["smplx"])
            if os.path.exists(path_smplx):
                from machx_utils.smplx import SmplxPara
                smplxpara = SmplxPara(smplxpara=path_smplx)
                direction, vector_direction, mark_direction = smplxpara.init_drn()
                annot["drn"] = direction
                annot["vec_drn"] = vector_direction
                annot["mark_drn"] = mark_direction

    
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


    

