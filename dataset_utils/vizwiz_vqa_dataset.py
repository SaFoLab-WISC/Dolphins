import sys

sys.path.append("..")

from dataset_utils.vqa_dataset import VQADataset
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import collate_fn

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import LlamaTokenizer, CLIPImageProcessor
from conversation import get_conv_template

'''
multimodal:
<image>User: what does the image describe? 
GPT:# two cats sleeping.<|endofchunk|>#
<image>User: what does the image describe? 
GPT:# a bathroom sink.<|endofchunk|>#
<image>User: what does the image describe? 
GPT:#{response}#

text:
User: what does the image describe? 
GPT:#two cats sleeping.<|endofchunk|>#

'''


class VizWizVQADataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, add_eos=True, ignore_instruction=True,
                 max_seq_length=512, max_num_images=5, mode="train", prompt_template="octopus"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images
        self.vis_root = vis_root
        self.prompt_template = prompt_template

        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                data = json.load(f)

        self.annotation = []
        for ex in data:
            image_id = ex['image']
            image_path = os.path.join(self.vis_root, image_id)
            assert os.path.exists(image_path)
            answers = ex['answers']
            answers = [item['answer'] for item in answers ]
            line = {}
            line['answer'] = max(set(answers), key=answers.count)
            line['question'] = ex['question']
            line['image_path'] = image_path
            self.annotation.append(line)

        self.vis_processor = DefaultTransform()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])


    def process_image(self, ann):
        image = Image.open(ann['image_path']).convert("RGB")
        image = [self.vis_processor(image)]
        region_images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]

        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros((self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size, self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        image = images_tensors.unsqueeze(1)

        return image

    def process_text(self, ann):
        conv = get_conv_template(self.prompt_template)
        conv.append_message(conv.roles[0], ann["question"])
        conv.append_message(conv.roles[1], ann["answer"])
        conversation = conv.get_prompt()

        return dict(conversation=conversation)

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True, max_length=self.max_seq_length, add_special_tokens=False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        return res

    def collater(self, samples):
        return collate_fn(samples, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id, left_pad=self.tokenizer.padding_side == "left")


from transformers import LlamaTokenizer, CLIPImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("anas-awadalla/mpt-7b")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]
    })

    image_processor = DefaultTransform()

    dataset = VizWizVQADataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/VizWiz/train",
        ann_paths=[
            "/home/yingzi/MultiInstruct/VizWiz/train.json"
        ],
        mode="train",
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    from tqdm import tqdm

    pbar = tqdm(total=len(train_dataloader))
    for batch in train_dataloader:
        batch = batch['net_input']
        pbar.update(1)

        images = batch['image']
        image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()

        start_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = 0
        labels[:, 0] = 0

        for i in range(labels.shape[0]):
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            media_idxs = torch.where(labels[i] == media_token_id)[0]

            # remove loss for any token the before the first <answer>
            token_idx = 0
            print(labels[i])
            if 32869 in labels[i]:
                while token_idx < labels.shape[1] and labels[i][token_idx] != 32869:
                    labels[i][token_idx] = 0
                    token_idx += 1
                labels[i][token_idx + 1] = 0
                labels[i][labels[i] == 32869] = 0
            elif 37741 in labels[i]:
                while token_idx < labels.shape[1] and labels[i][token_idx] != 37741:
                    labels[i][token_idx] = 0
                    token_idx += 1
                labels[i][token_idx + 1] = 0
                labels[i][labels[i] == 37741] = 0
            else:
                while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                    labels[i][token_idx] = 0
                    token_idx += 1

            # # remove loss for any token between <|endofchunk|> and <answer>, except <image>
            # for endofchunk_idx in endofchunk_idxs:
            #     token_idx = endofchunk_idx + 1
            #     while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
            #         if labels[i][token_idx] == media_token_id:
            #             pass
            #         else:
            #             labels[i][token_idx] = -100
            #         token_idx += 1

        labels[labels == answer_token_id] = 0
        labels[labels == media_token_id] = 0

        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        print(tokenizer.decode(labels[0][:sum(batch['attention_mask'][0])]))

