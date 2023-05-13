import torch
import torch.nn as nn
from einops import rearrange, repeat
from lavis.models import load_model_and_preprocess
from transformers import AutoModelForCausalLM
from .base_model import BaseModel


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()
        inner_dim = int(dim * ff_mult)
        self.norm = nn.LayerNorm(dim)
        self.w_1 = nn.Linear(dim, inner_dim)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)    
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, dkv):
        super().__init__()
        self.scale = dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dkv, dim, bias=False)
        self.to_v = nn.Linear(dkv, dim, bias=False)
        
        self.to_o = nn.Linear(dim, dim)

    def forward(self, x, latents):
        
        x = self.norm(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q * self.scale
        
        sim = q @ k.transpose(-2, -1)
        sim = sim.softmax(dim=-1)
        out = sim @ v
        
        return self.to_o(out)  


class Adapter(nn.Module):
    def __init__(self, in_dim, dim, heads, dim_head, ff_mult=4, num_latents=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.atten = Attention(dim=dim, dkv=in_dim)
        self.ffnet = FeedForward(dim=dim, ff_mult=ff_mult)
        
    def forward(self, x):
        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])
        
        latents = self.atten(x, latents) + latents
        latents = self.ffnet(latents) + latents
        
        return latents


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
    def __init__(self, model_name, tokenizer, vision_token='<img>', cache_dir=None):
        super().__init__()
        self.vision = VisionEncoder()
        if 'mpt' in model_name:
            self.lm = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        dim_v = self.vision.QFormer.vision_proj.in_features
        dim_l = self.lm.get_input_embeddings().embedding_dim
        self.vision_lang_adapter = Adapter(dim_v, dim_l, depth=1, heads=12, dim_head=64, ff_mult=4, num_latents=1)
        self.vision_token = vision_token
        self.vision_token_id = tokenizer.convert_tokens_to_ids(vision_token)
        
        self.freeze_vision_and_lm()
        
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
            
        if vision_x is None:
            with self.maybe_autocast():
                outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs
        else:
            inputs_embeds = self.ids_to_tokens(input_ids) # (B, L, D)
            
            B, N, C, H, W = vision_x.shape
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, 1, inputs_embeds.shape[-1], device=self.device)], dim=1)
            
            vision_x = rearrange(vision_x, 'b n c h w -> (b n) c h w')
            vision_embeds = self.vision(vision_x)
            vision_embeds = self.vision_lang_adapter(vision_embeds)
            vision_embeds = rearrange(vision_embeds, '(b n) l d -> b (n l) d', b=B)
            # vision position
            vision_position = self.get_vision_position(input_ids)
            vision_position = vision_position[:, :N]
            pos_x = vision_position
            pos_y = repeat(torch.arange(B), 'b -> b n', n=vision_position.shape[-1])
            # insert vision embeds into inputs_embeds
            inputs_embeds = inputs_embeds.index_put((pos_y, pos_x), vision_embeds)
            inputs_embeds = inputs_embeds[:, :-1]
        
            with self.maybe_autocast():
                outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

            return outputs
