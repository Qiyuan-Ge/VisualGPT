import torch
from typing import List
from .helper import get_vision_position


class Assistant:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vision_token = "<img>"
    
    @torch.no_grad()
    def response(
        self,
        prompts: List[str],
        vision_x=None, 
        max_gen_len: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[str]:
        bsz = len(prompts)
        
        tokenized = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = tokenized['input_ids']
        
        min_prompt_size = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).min().item()
        max_prompt_size = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).max().item()
        
        total_len = min(self.tokenizer.model_max_length, max_gen_len + max_prompt_size)
        
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.model.device)
        tokens[:, :max_prompt_size] = input_ids
        
        input_text_mask = tokens != self.tokenizer.pad_token_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model(tokens[:, :cur_pos], vision_x=vision_x)
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[:input_text_mask[i].sum().item()+max_gen_len]
            # cut to eos tok if any
            try:
                if t[0] == self.tokenizer.eos_token_id:
                    t = t[1:]
                t = t[:t.index(self.tokenizer.eos_token_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(torch.tensor(t), skip_special_tokens=True))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token                
