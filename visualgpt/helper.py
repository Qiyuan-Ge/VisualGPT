import os
import torch
from einops import repeat
from omegaconf import OmegaConf
from lavis.models import load_preprocess

cur_dir = os.path.dirname(os.path.abspath(__file__))

def blip2_vision_processor(model='eval'):
    cfg = OmegaConf.load(os.path.join(cur_dir, "blip2_pretrain.yaml"))
    vis_processors, _ = load_preprocess(cfg.preprocess)
    
    return vis_processors[model]

# def get_vision_position(input_ids, vision_token_id):
#     mask_value = input_ids.shape[1]
#     vision_position = torch.arange(input_ids.shape[-1])
#     vision_position = repeat(vision_position, 'n -> b n', b=input_ids.shape[0])
#     vision_position = vision_position.masked_fill(input_ids!=vision_token_id, mask_value).sort(dim=-1).values
#     return vision_position
