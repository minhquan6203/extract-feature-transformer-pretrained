from typing import Text
import torch
import time
from data_utils.load_data import Data_Loader
from text_module.text_embedding import Text_Embedding
import numpy as np
import os
class Extract_Text():
    def __init__(self,config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_loader=Data_Loader(config)
        self.text_embedding=Text_Embedding(config).to(self.device)
        self.folder_out=config['text_folder_out']
    def extract(self):
        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)
        print("loading data")
        data = self.data_loader.get_dataloader()
        total_time = 0 
        print("extracting, please wait")
        start_time = time.time()
        for item in data:
            item_start_time = time.time()
            feature, mask = self.text_embedding(item['text'])
            for i in range(len(item['id'])):
                np.save(
                    os.path.join(self.folder_out,f"{item['id'][i]}.npy"),
                    {
                      "id": item["id"][i],
                      "text_feature": feature[i].detach().cpu().numpy(),
                      "text_mask": mask[i].detach().cpu().numpy()
                    },
                    allow_pickle = True
                )

            item_end_time = time.time()
            item_time = item_end_time - item_start_time
            print(f"Time taken for item: {item_time} seconds")
            total_time += item_time
        end_time = time.time()
        total_execution_time = end_time - start_time
        print(f"Total time taken: {total_execution_time} seconds")
        print(f"Average time per item: {total_time / len(data)} seconds")
