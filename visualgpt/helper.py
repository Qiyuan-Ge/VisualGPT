import torch
from transformers import BlipImageProcessor

class ImageProcessor:
    def __init__(self, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], size={'height': 224, 'width': 224}):
        self.processor = BlipImageProcessor(image_mean=mean, image_std=std, size=size)
        
    def __call__(self, image):
        image = self.processor(image)['pixel_values'][0]
        image = torch.tensor(image)
        
        return image
