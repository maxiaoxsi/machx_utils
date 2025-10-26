class Dataset:
    def __init__(self, jsons):
        jsons_dict = {}
        for json in jsons:
            datasetid = json.get_datasetid()
            if datasetid not in self._jsons:
                jsons_dict[datasetid] = [json]
            else:
                jsons_dict[datasetid].append(json)

        for key, item in jsons_dict:
            print(key)
            print(len(item))


    def __len__(self):
        return 


    def __getitem__(self, idx):
        pass