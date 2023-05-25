from typing import Text
import torch
import time
import os
import np
from data_utils.load_data import Data_Loader
from vision_module.vision_embedding import Vision_Embedding

class Extract_Image():
    def __init__(self,config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_loader=Data_Loader(config)
        self.vision_embedding=Vision_Embedding(config)
        data = self.data_loader.get_dataloader()
        self.folder_out=config['image_folder_out']
        total_time = 0 
        print("extracting, please wait")
        start_time = time.time()
        for item in data:
            item_start_time = time.time()
            feature, mask = self.vision_embedding(item['text'])
            np.save(
                os.join(self.folder_out,f"{item['id']}.npy"),
                {
                    "id":item["id"],
                    "text_feature":feature,
                    "text_mask":mask
                }
            )

            item_end_time = time.time()
            item_time = item_end_time - item_start_time
            print(f"Time taken for item: {item_time} seconds")
            total_time += item_time
        end_time = time.time()
        total_execution_time = end_time - start_time
        print(f"Total time taken: {total_execution_time} seconds")
        print(f"Average time per item: {total_time / len(data)} seconds")

