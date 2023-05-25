from torch.utils.data import DataLoader
from typing import Text, Dict, List
import json

class Data_Loader:
    def __init__(self,config: Dict):
        self.json_file=config['json_file']
        self.batch_size=config['batch_size']
        self.num_workers=config['num_workers']
    def load_json(self,json_file):
        with open(json_file) as f:
          data=json.load(f)
        annotations=[]
        for k,v in data.items():
            ann={
                'id':k,
                'text':v,
            }
            annotations.append(ann)
        return annotations

    def get_dataloader(self):
        dataset=self.load_json(self.json_file)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return dataloader