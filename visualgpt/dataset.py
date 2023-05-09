import torch
import copy
import json
import zipfile
import braceexpand
from PIL import Image


class MMC4:
    def __init__(self, input_shards, image_folder, vision_token='<img>', vision_processor=None):
        self.vision_token = vision_token
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        self.samples = []
        doc_shards = list(braceexpand.braceexpand(input_shards))
        for i in range(len(doc_shards)):
            with zipfile.ZipFile(doc_shards[i], "r") as zip_file:
                json_filename = zip_file.namelist()[0]
                with zip_file.open(json_filename, "r") as json_file:
                    for sample_data in json_file:
                        sample_data = json.loads(sample_data)
                        sample_data['shard_id'] = i
                        self.samples.append(sample_data)
            
    def __getitem__(self, idx):
        sample = self.samples[idx]
        lang_x = copy.deepcopy(sample['text_list'])
        vision_x = []
        shard_id = sample['shard_id']
        for image_info in sample["image_info"]:
            image_name = image_info['image_name']
            text_index = image_info['matched_text_index']
            try:
                img_path = f"{self.image_folder}/{shard_id}/{image_name}"
                img = Image.open(img_path).convert('RGB')
                img = self.vision_processor(img)
                vision_x.append(img.unsqueeze(0))
                lang_x[text_index] = self.vision_token + ' ' + lang_x[text_index]
            except:
                pass
        
        lang_x = ' '.join(lang_x)
        
        if len(vision_x) == 0:
            vision_x = torch.zeros(1, 3, 224, 224)
        else:
            vision_x = torch.cat(vision_x, dim=0)
        
        return dict(input_ids=lang_x, vision_x=vision_x)
    
    def __len__(self):
        return len(self.samples)