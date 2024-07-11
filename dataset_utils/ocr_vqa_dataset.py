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

class OCRVQADataset(VQADataset):
    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = [self.vis_processor(image)]

        region_images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]

        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        image = images_tensors.unsqueeze(1)
        return image

    def process_text(self, anns):
        # TODO remove this
        num_convs = len(anns["questions"])
        questions, answers = [], []

        for conv_id in range(0, num_convs):
            questions.append(anns["questions"][conv_id])
            answers.append(anns["answers"][conv_id])

        conversation = self.prompter(question=questions, response=answers, num_images=1, num_convs=len(questions))
        return dict(conversation=conversation)

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True, max_length=self.max_seq_length, add_special_tokens=False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        return res

    def collater(self, samples):

        return collate_fn(samples,
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id,
                          left_pad=self.tokenizer.padding_side == "left"
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

    image_processor = RegionTransform()

    dataset = OCRVQADataset(
        tokenizer=tokenizer,
        vis_processor=image_processor,
        vis_root="/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/OCR-VQA/images_tmp",
        ann_paths=
        ["/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/OCR-VQA/ocr_vqa.jsonl",
         ]
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        #sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    for batch in train_dataloader:
        batch = batch['net_input']
        for k, v in batch.items():
            if isinstance(v, list):
                print(k, len(v))
                continue
            print(k, v.shape)

        images = batch['image']
        image = images.squeeze(2)[1][0].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/ocr.png")
        print(batch["input_ids"][1][:sum(batch['attention_mask'][1])])
        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        break