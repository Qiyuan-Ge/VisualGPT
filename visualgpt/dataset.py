import copy
import json
import torch
import logging
import zipfile
import braceexpand
import transformers
from PIL import Image
from typing import Dict, Sequence
from torch.utils.data import Dataset

IGNORE_INDEX = -100
VISION_TOKEN = '<img>'

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

CAPTION_TEMPLATE = [
    "Describe this picture",
    "Can you provide a description of the picture?",
    "Can you provide a description of the scene captured in this picture?"
    "What can you tell me about the visual elements captured in this image?"
    "Please offer a thorough depiction of the scene depicted in this picture"
    "Could you narrate the contents of this picture?"
]


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# class COCOImageCaption(Dataset):


class VQA2(Dataset):
    def __init__(self, ann_folder, image_folder, tokenizer, ann_type='train', vision_processor=None, vision_token='<img>'):
        super().__init__()
        self.ann_type = ann_type
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        self.samples = []
        
        anno = json.load(open(f'{ann_folder}/v2_mscoco_{ann_type}2014_annotations.json', 'r'))
        ques = json.load(open(f'{ann_folder}/v2_OpenEnded_mscoco_{ann_type}2014_questions.json', 'r'))
        
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = []
        targets = []
        self.img_ids = []
        
        for i in range(len(anno['annotations'])):
            ans = anno['annotations'][i]['multiple_choice_answer']
            question_id = anno['annotations'][i]['question_id']
            image_id = anno['annotations'][i]['image_id']
            image_id = '0' * (12 - len(str(image_id))) + str(image_id)
            if question_id != ques['questions'][i]['question_id']:
                raise ValueError("question_id doesn't match")
            question = f"{vision_token} {ques['questions'][i]['question']}"
            example = {'ques_id': question_id, 'img_id': image_id, 'instruction': question, 'output': ans}
            example['instruction'] = prompt_no_input.format_map(example)
            
            sources.append(example['instruction'])
            targets.append(f"{example['output']}{tokenizer.eos_token}")
            self.img_ids.append(example['img_id'])
            
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
          
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = f"{self.image_folder}/{self.ann_type}2014/COCO_{self.ann_type}2014_{img_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = self.vision_processor(img)
        img = img.unsqueeze(0)
        
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], vision_x=img)
    
    def __len__(self):
        return len(self.input_ids)
        
        
class MMC4(Dataset):
    def __init__(self, input_shards, image_folder, vision_processor=None, vision_token='<img>'):
        super().__init__()
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
