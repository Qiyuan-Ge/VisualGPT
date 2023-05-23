import torch
import torch.nn as nn
from einops import rearrange, repeat
from lavis.models import load_model_and_preprocess
from transformers import AutoModelForCausalLM
from .base_model import BaseModel


class VisionEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.QFormer, _, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
        
    def forward(self, vision_input):
        """_summary_

        Args:
            vision_input (torch.Tensor): (B, N, C, H, W)
        """        
        
        sample = {"image": vision_input, "text_input": []}
        output = self.QFormer.extract_features(sample, mode="image")
        image_embeds = output.image_embeds
        return image_embeds
        

class VisualLM(BaseModel):
    def __init__(self, model_name, cache_dir=None):
        super().__init__()
        self.vision = VisionEncoder()
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        dim_v = self.vision.QFormer.vision_proj.in_features
        dim_l = self.lm.get_input_embeddings().embedding_dim
        self.norm = nn.LayerNorm(dim_v)
        self.proj = nn.Linear(dim_v, dim_l)
        self.vision_token_id = None
        self.freeze_vision_and_lm()
        
    def set_vision_token_id(self, tokenizer, vision_token='<img>'):
        self.vision_token_id = tokenizer.convert_tokens_to_ids(vision_token)
        
    def freeze_vision_and_lm(self):
        for param in self.vision.parameters():
            param.requires_grad = False
            
        for param in self.lm.parameters():
            param.requires_grad = False
    
    def ids_to_tokens(self, ids):
        return self.lm.get_input_embeddings()(ids)
    
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
        inputs_embeds = self.ids_to_tokens(input_ids)
        
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
        
        with self.maybe_autocast():
            outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        return outputs
