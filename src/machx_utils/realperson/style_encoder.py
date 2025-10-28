from machx_utils.realperson import RealPersonJson
import random

class StyleJson(RealPersonJson):
    def __init__(self, dirname, subdataset) -> None:
        super().__init__(dirname, subdataset)

    def get_cams(self):
        cams = {}
        for img in self._json['images']:
            camid = img['camid']
            imgid = img["id"]
            if camid not in cams:
                cams[camid] = [(self, imgid)]
            else:
                cams[camid].append((self, imgid))
        return cams



class StyleDataset:
    def __init__(self, jsons):
        self._doms = {}
        for json in jsons:
            datasetid = json.get_datasetid()
            if datasetid not in self._doms:
                self._doms[datasetid] = {}
            cams = json.get_cams()
            for camid, cam in cams.items():
                if camid not in self._doms[datasetid]:
                    self._doms[datasetid][camid] = cam
                else:
                    self._doms[datasetid][camid] = self._doms[datasetid][camid] + cam

    
    def __getitem__(self, idx):
        datasetids = random.sample(list(self._doms.keys()), 2)
        dataset1 = self._doms[datasetids[0]]
        dataset2 = self._doms[datasetids[1]]
        img_d1c1, img_d1c2 = self._get_imgs_from_dataset(dataset1)
        img_d2c1, _ = self._get_imgs_from_dataset(dataset2)
        return img_d1c1, img_d1c2, img_d2c1


    def _get_imgs_from_dataset(self, dataset):
        camids = random.sample(list(dataset.keys()), 2)
        cam1 = dataset[camids[0]]
        cam2 = dataset[camids[1]]
        return self._get_img_from_cam(cam1), self._get_img_from_cam(cam2)

    
    def _get_img_from_cam(self, imgs_list):
        json, imgid = random.sample(imgs_list, 1)[0]
        path_img = json.get_path("reid", imgid, is_img=True)
        return path_img
        

    def __len__(self):
        return 10000

