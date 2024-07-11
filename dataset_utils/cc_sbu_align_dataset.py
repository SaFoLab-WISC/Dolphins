import sys

sys.path.append("..")

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from PIL import Image
import  matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from transformers import LlamaTokenizer, CLIPImageProcessor

from dataset_utils.vqa_dataset import VQADataset, VisualPrompter
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn
from conversation import get_conv_template

QUESTIONS = [
        "Describe the following image in detail.",
        "Provide a detailed description of the given image.",
        "Give an elaborate explanation of the image you see.",
        "Share a comprehensive rundown of the presented image.",
        "Offer a thorough analysis of the image.",
        "Explain the various aspects of the image before you.",
        "Clarify the contents of the displayed image with great detail.",
        "Characterize the image using a well-detailed description.",
        "Break down the elements of the image in a detailed manner.",
        "Walk through the important details of the image.",
        "Portray the image with a rich, descriptive narrative.",
        "Narrate the contents of the image with precision.",
        "Analyze the image in a comprehensive and detailed manner.",
        "Illustrate the image through a descriptive explanation.",
        "Examine the image closely and share its details.",
        "Write an exhaustive depiction of the given image.",
]


class CcSbuAlignDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, add_eos=True, ignore_instruction=True, max_seq_length=512, max_num_images=5,):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))["annotations"])

        self.vis_processor = DefaultTransform()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

 
    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["image_id"] + ".jpg")
        image = Image.open(image_path).convert("RGB")
        image = [self.vis_processor(image)]

        images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(images), self.max_num_images))
        images_tensors = images[keep_ixs]

        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

        image = images_tensors.unsqueeze(1)
        return image

    def process_text(self, ann):
        image_caption = ann["caption"]
        instruction = random.choice(QUESTIONS)
        conv = get_conv_template("octopus")
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], image_caption)
        conversation = conv.get_prompt()
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

    dataset = CcSbuAlignDataset(
        tokenizer=tokenizer,
        vis_processor=image_processor,
        vis_root="/home/yingzi/vlm/workspace/dataset/cc_sbu_align/image",
        ann_paths=["/home/yingzi/vlm/workspace/dataset/cc_sbu_align/filter_cap.json",
                   ],
        max_seq_length=256,
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
        image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.savefig("/home/yingzi/vlm/examples/minigpt-4.png")

        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        break
