import torch
import torch.nn as nn
import transformers
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from transformers import Blip2VisionConfig, Blip2QFormerConfig


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = transformers.Blip2VisionModel(Blip2VisionConfig())
        self.num_query_tokens = 32
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_query_tokens, Blip2QFormerConfig().hidden_size))
        self.qformer = transformers.Blip2QFormerModel(Blip2QFormerConfig())
        
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

    
class VisualLM(nn.Module):
    def __init__(self, language_model_name="RootYuan/RedLing-7B-v0.1", cache_dir=None):
        super().__init__()
        self.vision = Vision()
        self.lm = transformers.AutoModelForCausalLM.from_pretrained(language_model_name, cache_dir=cache_dir)
        self.config = self.lm.config
        dim_l = self.lm.get_input_embeddings().embedding_dim
        dim_q = 768
        self.norm = nn.LayerNorm(dim_q)
        self.proj = nn.Linear(dim_q, dim_l)
        self.vision_token_id = None
        
    def load_pretrained_vision(self, checkpoint_path=None, cache_dir=None):
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("RootYuan/blip2_vision_model", "pytorch_model.bin", cache_dir=cache_dir)
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
        vision_position = torch.arange(input_ids.shape[-1]).to(input_ids.device)
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
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, 1, inputs_embeds.shape[-1]).type_as(inputs_embeds)], dim=1)
            
            vision_x = rearrange(vision_x, 'b n c h w -> (b n) c h w')
            vision_embeds = self.vision(vision_x)
            vision_embeds = rearrange(vision_embeds, '(b n) l d -> b (n l) d', b=B)
            vision_embeds = self.norm(vision_embeds)
            vision_embeds = self.proj(vision_embeds)
            # vision position
            vision_position = self.get_vision_position(input_ids)
            vision_position = vision_position[:, :vision_embeds.shape[-2]]
            pos_x = vision_position
            pos_y = repeat(torch.arange(B), 'b -> b n', n=vision_position.shape[-1]).type_as(pos_x)
            # insert vision embeds into inputs_embeds
            inputs_embeds = inputs_embeds.index_put((pos_y, pos_x), vision_embeds)
            inputs_embeds = inputs_embeds[:, :-1]
        
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        return outputs
    
    @torch.no_grad()
    def generate(self, input_ids, vision_x=None, attention_mask=None, **generate_kwargs):

        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        if vision_x is not None:
            assert self.vision_token_id is not None, "Please set vision_token_id first"
            
            B, N, C, H, W = vision_x.shape
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, 1, inputs_embeds.shape[-1]).type_as(inputs_embeds)], dim=1)
            
            vision_x = rearrange(vision_x, 'b n c h w -> (b n) c h w')
            vision_embeds = self.vision(vision_x)
            vision_embeds = rearrange(vision_embeds, '(b n) l d -> b (n l) d', b=B)
            vision_embeds = self.norm(vision_embeds)
            vision_embeds = self.proj(vision_embeds)
            # vision position
            vision_position = self.get_vision_position(input_ids)
            vision_position = vision_position[:, :vision_embeds.shape[-2]]
            pos_x = vision_position
            pos_y = repeat(torch.arange(B), 'b -> b n', n=vision_position.shape[-1]).type_as(pos_x)
            # insert vision embeds into inputs_embeds
            inputs_embeds = inputs_embeds.index_put((pos_y, pos_x), vision_embeds)
            inputs_embeds = inputs_embeds[:, :-1]
        
        outputs = self.lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generate_kwargs)
        
        return outputs
