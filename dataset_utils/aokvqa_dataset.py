import sys

sys.path.append("..")

from dataset_utils.vqa_dataset import VQADataset
from dataset_utils.processors import DefaultTransform, RegionTransform
from dataset_utils.collater import  collate_fn

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
from transformers import LlamaTokenizer
REASON_QUESTIONS = [
    "Why?",
    "Why is this?",
    "And why?",
    "What is the reason?",
    "And can you tell me why?",
    "Can you tell me why?",
    "Can you tell me the reason?",
]


class AOKVQADataset(VQADataset):
    def __init__(self, tokenizer, vis_processor, vis_root, ann_paths, **kwargs):
        super().__init__(tokenizer, vis_processor, vis_root, ann_paths, **kwargs)

    def process_image(self, ann):
        image_file = "0" * (12 - len(str(ann["image_id"]))) +  str(ann["image_id"]) + ".jpg"
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = [self.vis_processor(image)]

        images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(images), self.max_num_images))
        images_tensors = images[keep_ixs]
        # pad to 5 images
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size, self.vis_processor.image_size),
                dtype=torch.float
            )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

        image = images_tensors.unsqueeze(1)
        return image

    def process_text(self, ann):
        question = ann["question"]
        question = question + " " + random.choice(REASON_QUESTIONS)

        choices = ann["choices"]
        true_answer = choices[ann["correct_choice_idx"]]
        answer = "The answer is " + true_answer + ". Because " + " ".join(ann["rationales"])

        is_option = random.random() < self.option_prob and len(choices) > 1
        if is_option:
            options = ", ".join(choices)
            src_text = f"<image> User: {question} context: {options} GPT:<answer> {answer} <|endofchunk|>"
        else:
            src_text = f"<image> User: {question} GPT:<answer> {answer} <|endofchunk|>"

        return dict(src_text=src_text)

    def tokenize(self, text):
        res = self.tokenizer(text["src_text"], return_tensors="pt", padding="do_not_pad", truncation=True,
                             max_length=self.max_seq_length, add_special_tokens=False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        return res

    def collater(self, samples):

        return collate_fn(samples,
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id,
                          left_pad=self.tokenizer.padding_side == "left"
                          )


def build_aokvqa_dataset(
    tokenizer,
    vis_processor,
    vis_root="data/coco/images",
    ann_paths=["data/aokvqa/annotations/aokvqa_v1p0_train.json"],
    sample_image=False,
):
    return AOKVQADataset(
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        vis_root=vis_root,
        ann_paths=ann_paths,
        sample_image=sample_image,
    )


from transformers import LlamaTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader
import  matplotlib.pyplot as plt


if __name__ == "__main__":

    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]
    })

    image_processor = DefaultTransform()

    dataset = AOKVQADataset(
        tokenizer=tokenizer,
        vis_processor=image_processor,
        vis_root="/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/COCO",
        ann_paths=
        ["/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/AOK-VQA/aokvqa_v1p0_train.json",
         ]
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        #sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    from tqdm import tqdm

    for batch in tqdm(train_dataloader):
        batch = batch['net_input']

    for batch in train_dataloader:
        batch = batch['net_input']
        for k, v in batch.items():
            if isinstance(v, list):
                print(k, len(v))
                continue
            print(k, v.shape)

        images = batch['image']
        plt.figure()
        image = images.squeeze(2)[1][0].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/aokvqa.png")

        print(batch["input_ids"][1][:sum(batch['attention_mask'][1])])
        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        break