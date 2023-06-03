from torch.utils.data import DataLoader
from typing import Text, Dict, List
import json
import os

class Data_Loader:
    def __init__(self,config: Dict):
        self.json_file=config['json_file']
        self.batch_size=config['batch_size']
        self.num_workers=config['num_workers']

    def load_annotations(self, json_file)-> List[Dict]:
        with open(json_file) as f:
            json_data =json.load(f)
        annotations = []
        for ann in json_data["annotations"]:
            question = ann["question"]
            answer = ann['answers'][0]
            image_id = ann["image_id"]
            id = ann['id']
            annotation = {
                "id": id,
                "question": question,
                "answer": answer,
                "image_id": image_id,
            }
            annotations.append(annotation)
        return annotations
   
    def get_dataloader(self):
        dataset=self.load_annotations(self.json_file)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
        return dataloader