import argparse
import torch
from utils.get_config import get_config
from text_module.extract_text import Extract_Text
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required = True)

args = parser.parse_args()

config = get_config(args.config_file)
Extract_Text(config).extract()
print("done")