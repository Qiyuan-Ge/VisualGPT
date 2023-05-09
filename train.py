import copy
import torch
import visualgpt
import transformers
from transformers import Trainer, AutoTokenizer, AddedToken
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from visualgpt.helper import blip2_vision_processor, get_vision_position
from visualgpt.dataset import MMC4


IGNORE_INDEX = -100
VISION_TOKEN = "<img>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default="facebook/opt-125m")
    
    
@dataclass
class DataArguments:
    input_shards: str = field(default=None, metadata={"help": "Path to the training doc shards."})
    image_folder: str = field(default=None, metadata={"help": "Path to the training image folder."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_steps: int = field(default=1)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer):
    tokenized = tokenizer(
        strings,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = tokenized.input_ids
    labels = copy.deepcopy(input_ids)
    ignore = (input_ids == tokenizer.pad_token_id) | (input_ids == tokenizer.convert_tokens_to_ids(VISION_TOKEN))
    labels = labels.masked_fill(ignore, IGNORE_INDEX)
    
    return dict(input_ids=input_ids, labels=labels, attention_mask=tokenized.attention_mask)


@dataclass
class DataCollatorForMutiModalDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances):
        input_ids, vision_x = tuple([instance[key] for instance in instances] for key in ("input_ids", "vision_x"))
        tokenized = _tokenize_fn(input_ids, self.tokenizer)
        
        vision_x = torch.nn.utils.rnn.pad_sequence(vision_x, batch_first=True, padding_value=0.0)
        vision_position = get_vision_position(tokenized['input_ids'], self.tokenizer.convert_tokens_to_ids(VISION_TOKEN))
        
        return dict(
            input_ids=tokenized['input_ids'], 
            vision_x=vision_x,
            vision_position=vision_position,
            attention_mask=tokenized['attention_mask'],
            labels=tokenized['labels'], 
        )


def make_unsupervised_data_module(tokenizer, data_args):
    train_dataset = MMC4(data_args.input_shards, data_args.image_folder, vision_token='<img>', vision_processor=blip2_vision_processor())
    data_collator = DataCollatorForMutiModalDataset(tokenizer)
    
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    other_tokens,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    for new_token in other_tokens:
        num_new_tokens += tokenizer.add_tokens(AddedToken(new_token, normalized=False))

    model.lm.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.lm.get_input_embeddings().weight.data
        output_embeddings = model.lm.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if 'stablelm' in model_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    model = visualgpt.VisualLM(model_name=model_args.model_name)
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    other_tokens = [VISION_TOKEN]    
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        other_tokens=other_tokens,
        tokenizer=tokenizer,
        model=model,
    )
    
    model.freeze_vision_and_lm()
    
    data_module = make_unsupervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()