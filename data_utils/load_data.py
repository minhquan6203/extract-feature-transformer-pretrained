from torch.utils.data import DataLoader
from typing import Text, Dict, List
import json

class Data_Loader:
    def __init__(self,config: Dict):
        self.json_file=config['json_file']
        self.batch_size=config['batch_size']
    def load_json(self,json_file):
        data=json.load(json_file)
        annotations=[]
        for k,v in data.items():
            ann={
                'id':k,
                'text':v,
            }
            annotations.append(ann)
        return annotations

    def get_dataloader(self):
        dataset=self.load_test_set(self.json_file)
        dataloader = DataLoader(
            dataset,
            batch_size=self.bath_size,
            num_workers=4,
        )
        return dataloader