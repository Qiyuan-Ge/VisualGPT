import json
import torch
import torch.nn as nn
import transformers
from copy import deepcopy
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from transformers import Blip2VisionConfig, Blip2QFormerConfig

from .base_model import BaseModel
# from lavis.models import load_model_and_preprocess
# class VisionEncoder(nn.Module):
#     def __init__(self, device='cpu'):
#         super().__init__()
#         self.QFormer, _, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        
#     def forward(self, vision_input):
#         """_summary_

#         Args:
#             vision_input (torch.Tensor): (B, N, C, H, W)
#         """        
        
#         sample = {"image": vision_input, "text_input": []}
#         output = self.QFormer.extract_features(sample, mode="image")
#         image_embeds = output.image_embeds
#         return image_embeds

   
class Config:
    def __init__(self):
        self.num_query_tokens = 32
        self.query_dim = 768
        self.vision_config = Blip2VisionConfig()
        self.qformer_config = Blip2QFormerConfig()
        self.language_model_name = "RootYuan/RedLing-7B-v0.1"
        self.cache_dir = None
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        return deepcopy(self.__dict__)
    
    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)
            
    def from_dict(self, dct):
        self.clear()
        for key, value in dct.items():
            self.__dict__[key] = value
            
        return self.to_dict()


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = transformers.Blip2VisionModel(config.vision_config)
        
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = transformers.Blip2QFormerModel(config.qformer_config)
        
    def forward(self, vision_x):
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values=vision_x)
        image_embeds = vision_outputs[0]
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs[0]
        
        return query_output

    
class VisualLM(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = Vision(config)
        self.lm = transformers.AutoModelForCausalLM.from_pretrained(config.language_model_name, cache_dir=config.cache_dir)
        dim_l = self.lm.get_input_embeddings().embedding_dim
        self.norm = nn.LayerNorm(config.query_dim)
        self.proj = nn.Linear(config.query_dim, dim_l)
        self.vision_token_id = None
        
    def load_pretrained_vision(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("RootYuan/blip2_vision_model", "pytorch_model.bin", cache_dir=self.config.cache_dir)
        self.vision.load_state_dict(torch.load(checkpoint_path))
    
    def freeze_vision(self):
        for param in self.vision.parameters():
            param.requires_grad = False
        
    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def set_vision_token_id(self, tokenizer, vision_token='<img>'):
        self.vision_token_id = tokenizer.convert_tokens_to_ids(vision_token)
    
    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()
    
    def get_vision_position(self, input_ids):
        mask_value = input_ids.shape[1]
        vision_position = torch.arange(input_ids.shape[-1]).to(self.device)
        vision_position = repeat(vision_position, 'n -> b n', b=input_ids.shape[0])
        vision_position = vision_position.masked_fill(input_ids!=self.vision_token_id, mask_value).sort(dim=-1).values
        return vision_position
        
    def forward(self, input_ids, vision_x=None, attention_mask=None, labels=None):
        """_summary_

        Args:
            input_ids (torch.Tensor): (B, L)
            vision_x (torch.Tensor): (B, N, C, H, W)
        """        
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if vision_x is not None:
            assert self.vision_token_id is not None, "Please set vision_token_id first"
            
            B, N, C, H, W = vision_x.shape
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, 1, inputs_embeds.shape[-1], device=inputs_embeds.device)], dim=1)
            
            vision_x = rearrange(vision_x, 'b n c h w -> (b n) c h w')
            vision_embeds = self.vision(vision_x)
            vision_embeds = rearrange(vision_embeds, '(b n) l d -> b (n l) d', b=B)
            vision_embeds = self.norm(vision_embeds)
            vision_embeds = self.proj(vision_embeds)
            # vision position
            vision_position = self.get_vision_position(input_ids)
            vision_position = vision_position[:, :vision_embeds.shape[-2]]
            pos_x = vision_position
            pos_y = repeat(torch.arange(B), 'b -> b n', n=vision_position.shape[-1])
            # insert vision embeds into inputs_embeds
            inputs_embeds = inputs_embeds.index_put((pos_y, pos_x), vision_embeds)
            inputs_embeds = inputs_embeds[:, :-1]
        
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        return outputs
    
    @torch.no_grad()
    def generate(self, input_ids, vision_x=None, attention_mask=None, **generate_kwargs,):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if vision_x is not None:
            assert self.vision_token_id is not None, "Please set vision_token_id first"
            
            B, N, C, H, W = vision_x.shape
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, 1, inputs_embeds.shape[-1], device=self.device)], dim=1)
            
            vision_x = rearrange(vision_x, 'b n c h w -> (b n) c h w')
            vision_embeds = self.vision(vision_x)
            vision_embeds = rearrange(vision_embeds, '(b n) l d -> b (n l) d', b=B)
            vision_embeds = self.norm(vision_embeds)
            vision_embeds = self.proj(vision_embeds)
            # vision position
            vision_position = self.get_vision_position(input_ids)
            vision_position = vision_position[:, :vision_embeds.shape[-2]]
            pos_x = vision_position
            pos_y = repeat(torch.arange(B), 'b -> b n', n=vision_position.shape[-1])
            # insert vision embeds into inputs_embeds
            inputs_embeds = inputs_embeds.index_put((pos_y, pos_x), vision_embeds)
            inputs_embeds = inputs_embeds[:, :-1]
        
        outputs = self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        
        return outputs
