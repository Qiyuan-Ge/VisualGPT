import torch
import torch.nn as nn
import contextlib

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()