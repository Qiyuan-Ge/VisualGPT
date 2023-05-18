import os
import copy
import json
import torch
import logging
import zipfile
import numpy as np
import pandas as pd
import braceexpand
import transformers
from PIL import Image
from typing import Dict, Sequence
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from .utils import jload, jlload

IGNORE_INDEX = -100
VISION_TOKEN = '<img>'
VISION_TOKENS = VISION_TOKEN*32 + '\n'


DEFAULT_PROMPT_DICT = {
    "prompt_input": "<USER>:{user}\n<INPUT>:{input}\n<ASSISTANT>:",
    "prompt_no_input": "<USER>:{user}\n<ASSISTANT>:",
}


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{user}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{user}\n\n### Response:"
    ),
}


PROMPT_TEMPLATE = DEFAULT_PROMPT_DICT


CAPTION_TEMPLATE = [
    "Describe this picture",
    "Explain what you see in this image.",
    "Provide a detailed description of the picture.",
    "Can you describe the visual elements captured in this photograph?",
    "Please narrate the scene depicted in this image.",
    "Offer a verbal portrayal of what is happening in this picture.",
    "Can you provide a description of the picture?",
    "Can you provide a description of the scene captured in this picture?",
    "What can you tell me about the visual elements captured in this image?",
    "Please offer a thorough depiction of the scene depicted in this picture",
    "Could you narrate the contents of this picture?",
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


class ShareGPTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Tokenizing inputs... This may take some time...")
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        self.input_ids = []
        self.labels = []
        model_max_length = tokenizer.model_max_length
        
        for i in range(len(list_data_dict)):
            conversations = list_data_dict[i]['conversations']
            if len(conversations) < 2:
                continue
            sources = []
            targets = []
            next_speak = 'human'
            for example in conversations:
                if example['from'] == 'human':
                    if next_speak == 'human':
                        sources.append(PROMPT_NO_INPUT.format(user=example['value']))
                        next_speak = 'gpt'
                    else:
                        sources = targets = []
                        break
                elif example['from'] == 'gpt':
                    if next_speak == 'gpt':
                        targets.append(f"{example['value']}{tokenizer.eos_token}")
                        next_speak = 'human'
                    else:
                        sources = targets = []
                        break
                else:
                    print(f"Invalid example type {example['from']}")
                    sources = targets = []
                    break

            if len(sources) == 0 or len(targets) == 0 or len(sources) != len(targets):
                pass
            else:
                data_dict = preprocess(sources, targets, tokenizer)
                input_ids_cat = torch.cat(data_dict["input_ids"], dim=0)
                labels_cat = torch.cat(data_dict["labels"], dim=0)
                if input_ids_cat.shape[0] > model_max_length:
                    input_ids_cat = input_ids_cat[:model_max_length]
                    labels_cat = labels_cat[:model_max_length]
                self.input_ids.append(input_ids_cat)
                self.labels.append(labels_cat)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class AlpacaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        
        PROMPT_INPUT, PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_input"], PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = [
            PROMPT_INPUT.format(user=example['instruction'], input=example['input']) if example.get("input", "") != "" else PROMPT_NO_INPUT.format(user=example['instruction'])
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        self.sources = sources
        self.targets = targets
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

class DollyDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        list_data_dict = jlload(data_path)

        logging.warning("Formatting inputs...")
        
        PROMPT_INPUT, PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_input"], PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = [
            PROMPT_INPUT.format(user=example['instruction'], input=example['context']) if example.get("context", "") != "" else PROMPT_NO_INPUT.format(user=example['instruction'])
            for example in list_data_dict
        ]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class GradeSchoolMathDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        df = pd.read_parquet(data_path)
        list_data_dict = df.to_dict('records')

        logging.warning("Formatting inputs...")
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = [
            PROMPT_NO_INPUT.format(user=example['INSTRUCTION']) for example in list_data_dict
        ]
        targets = [f"{example['RESPONSE']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
    
class CompetitionMathDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")
        df = pd.read_parquet(data_path)
        list_data_dict = df.to_dict('records')

        logging.warning("Formatting inputs...")
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = [
            PROMPT_NO_INPUT.format(user=example['instruction']) for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LLaVAInstruct150K(Dataset):
    def __init__(self, data_path: str, image_folder:str, tokenizer: transformers.PreTrainedTokenizer, dataType='train', vision_processor=None):
        super().__init__()
        self.dataType = dataType
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        
        logging.warning("Tokenizing inputs... This may take some time...")
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        self.input_ids = []
        self.labels = []
        self.img_ids = []
        
        for i in range(len(list_data_dict)):
            img_id = list_data_dict[i]['id']
            conversations = list_data_dict[i]['conversations']
            if len(conversations) < 2:
                continue
            sources = []
            targets = []
            next_speak = 'human'
            for example in conversations:
                if example['from'] == 'human':
                    if next_speak == 'human':
                        example['value'] = example['value'].replace("<image>", VISION_TOKENS)
                        sources.append(PROMPT_NO_INPUT.format(user=example['value']))
                        next_speak = 'gpt'
                    else:
                        sources = []
                        targets = []
                        break
                elif example['from'] == 'gpt':
                    if next_speak == 'gpt':
                        targets.append(f"{example['value']}{tokenizer.eos_token}")
                        next_speak = 'human'
                    else:
                        sources = []
                        targets = []
                        break
                else:
                    print(f"Invalid example type {example['from']}")
                    sources = []
                    targets = []
                    break
            
            if len(sources) == 0 or len(targets) == 0 or len(sources) != len(targets):
                pass
            else:
                data_dict = preprocess(sources, targets, tokenizer)
                input_ids_cat = torch.cat(data_dict["input_ids"], dim=0)
                labels_cat = torch.cat(data_dict["labels"], dim=0)
                if torch.unique(labels_cat).shape[0] == 1 or input_ids_cat.shape[0] > tokenizer.model_max_length:
                    pass
                else:
                    self.input_ids.append(input_ids_cat)
                    self.labels.append(labels_cat)
                    self.img_ids.append(img_id)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = f"{self.image_folder}/{self.dataType}2014/COCO_{self.dataType}2014_{img_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = self.vision_processor(img)
        img = img.unsqueeze(0)
        
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], vision_x=img) 
    

class COCOImageCaption(Dataset):
    def __init__(self, root, tokenizer, dataType='train', vision_processor=None):
        self.root = root
        self.img_dir = '{}/{}2017'.format(root, dataType)
        self.vision_processor = vision_processor
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = []
        targets = []
        
        annFile = '{}/annotations/captions_{}2017.json'.format(root, dataType)
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        for i in range(len(self.img_ids)):
            imgid = self.img_ids[i]
            annid = self.coco.getAnnIds(imgIds=imgid)
            assistant = np.random.choice(self.coco.loadAnns(annid))['caption']
            
            user = f"{VISION_TOKENS}{np.random.choice(CAPTION_TEMPLATE)}"
            example = {'user': user, 'assistant': assistant}
            example['user'] = PROMPT_NO_INPUT.format(user=example['user'])
            
            sources.append(example['user'])
            targets.append(f"{example['assistant']}{tokenizer.eos_token}")
        
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
            
    def __getitem__(self, idx):
        imgid = self.img_ids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        img = self.vision_processor(img)
        img = img.unsqueeze(0)
        
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], vision_x=img)
        
    def __len__(self):
        return len(self.img_ids)


class VQA2(Dataset):
    def __init__(self, ann_folder, image_folder, tokenizer, dataType='train', vision_processor=None):
        super().__init__()
        self.dataType = dataType
        self.image_folder = image_folder
        self.vision_processor = vision_processor
        self.samples = []
        
        anno = json.load(open(f'{ann_folder}/v2_mscoco_{dataType}2014_annotations.json', 'r'))
        ques = json.load(open(f'{ann_folder}/v2_OpenEnded_mscoco_{dataType}2014_questions.json', 'r'))
        
        PROMPT_NO_INPUT = PROMPT_TEMPLATE["prompt_no_input"]
        
        sources = []
        targets = []
        self.img_ids = []
        
        for i in range(len(anno['annotations'])):
            assistant = anno['annotations'][i]['multiple_choice_answer']
            question_id = anno['annotations'][i]['question_id']
            image_id = anno['annotations'][i]['image_id']
            image_id = '0' * (12 - len(str(image_id))) + str(image_id)

            user = f"{VISION_TOKENS}{ques['questions'][i]['question']}"
            example = {'ques_id': question_id, 'img_id': image_id, 'user': user, 'assistant': assistant}
            example['user'] = PROMPT_NO_INPUT.format(user=example['user'])
            
            sources.append(example['user'])
            targets.append(f"{example['assistant']}{tokenizer.eos_token}")
            self.img_ids.append(example['img_id'])
            
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
          
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = f"{self.image_folder}/{self.dataType}2014/COCO_{self.dataType}2014_{img_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = self.vision_processor(img)
        img = img.unsqueeze(0)
        
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx], vision_x=img)
    
    def __len__(self):
        return len(self.input_ids)
        
        
class MMC4(Dataset):
    def __init__(self, input_shards, image_folder, vision_processor=None):
        super().__init__()
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
                lang_x[text_index] = VISION_TOKENS + ' ' + lang_x[text_index]
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
