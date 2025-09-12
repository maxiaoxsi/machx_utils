from ntpath import dirname
import os
import json
from tqdm import tqdm


class Json:
    def __init__(self, dirname, datasetname, subdataset) -> None:
        self._dirname = dirname  
        self._datasetname = datasetname
        self._subdataset = subdataset
        self._jsonpath = os.path.join(self._dirname, self._datasetname, f"annot/{self._subdataset}.json")
        self._imagedir = os.path.join(self._dirname, self._datasetname, 'image', self._subdataset)
        self._data = None
        self.load_json()

    def load_json(self):
        if not os.path.exists(self._jsonpath):
            print("json file not exists!")
            return
        with open(self._jsonpath, 'r', encoding='utf-8') as f:
            self._data = json.load(f)

    def save_json(self):
        with open(f"{self._jsonpath}_new", 'w', encoding='utf-8') as f:
            print(f"file saved in {self._jsonpath}_new")
            json.dump(self._data, f, indent=4)

    def check_image(self, i):
        if not self._data:
            self.load_json()
        print(self._data["images"][i])

    def init_annot(self):
        if os.path.exists(self._jsonpath):
            print("file exists!")
        self._data = self.make_json() 
        
    def init_json(self):
        if not self._data:
            self._data = {
                "info": {
                    "year": 2025,
                    "version": "1.0",
                    "datsetname": f"{self._datasetname}",
                    "subdataset": f"{self._subdataset}",
                    "description": f"RealPerson {self._datasetname} {self._subdataset}.",
                },
                "licenses": [],
                "images": [],
                "categories": [],
                "licenses": [],
                "annotations": []
            }
            self.save_json()

    def init_categories(self):
        list_personid = []
        id = 0
        for image in self._data['images']:
            personid = image['personid']
            if personid not in list_personid:
                list_personid.append(personid)
                item = {
                    "id": id,
                    "name": personid,
                    "supercategory": "-1",
                }
                self._data["categories"].append(item)
                id = id + 1
        self.save_json()

    def init_annotations(self):
        id = 0
        for image in self._data['images']:

            path_skeleton = os.path.join('./', self._datasetname, 'skeleton', self._subdataset, image['filename'])
            path_render = os.path.join('./', self._datasetname, 'render', self._subdataset, image['filename'])
            path_smplx = os.path.join('./', self._datasetname, 'smplx', self._subdataset, image['filename'])
            path_smplx = path_smplx.replace(path_smplx.split('.')[-1], 'npz')
            item = {}
            item["id"] = id
            if not os.path.exists(path_skeleton):
                item["skeleton"] = "-1"
            else:
                dir_skeleton = os.path.join('./', self._datasetname, 'skeleton', self._subdataset)
                item["skeleton"] = path_skeleton[len(dir_skeleton) + 1:]
            if not os.path.exists(path_render):
                item["render"] = "-1"
            else:
                dir_render = os.path.join('./', self._datasetname, 'render', self._subdataset)
                item["render"] = path_render[len(dir_render) + 1:]
            if not os.path.exists(path_smplx):
                item["smplx"] = "-1"
            else:
                dir_smplx = os.path.join('./', self._datasetname, 'smplx', self._subdataset)
                item["smplx"] = path_smplx[len(dir_smplx) + 1:]
            id = id + 1
            self._data["annotations"].append(item)
        self.save_json()

    def get_path(self, tgtdir, filename, ext):
        path = os.path.join(self._dirname, self._datasetname, tgtdir, self._subdataset, filename)
        if ext:
            if '.' in filename:
                path = os.path.splitext(path)[0]
            if '.' in ext:
                ext = ext[1:]
            path = f"{path}.{ext}"
        return path

    def process_batch(self, process_method, tgtdir, batch_size):
        data_batch = {
            "image_list":[],
            "tgt_list":[],
        }
        for image in tqdm(self._data["images"]):
            data_batch["image_list"].append(self.get_path("image", image["filename"], None))
            data_batch["tgt_list"].append(self.get_path(tgtdir, image["filename"], 'npy'))
            if len(data_batch["image_list"]) == batch_size:
                process_method(data_batch)
                data_batch["image_list"] = []
                data_batch["tgt_list"] = []
        process_method(data_batch)
        
        
if __name__ == '__main__':
    json_realperson = RealPersonJson('./', 'market1501', 'bounding_box_train')
    json_realperson.check_image(0)
    # json_realperson.init_annotations()