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


class LRVDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, add_eos=True, ignore_instruction=True,
                 max_seq_length=512, max_num_images=5, ):
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
        try:
            image_path = os.path.join(self.vis_root, "VG_100K", str(ann["image_id"]) + ".jpg")
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = os.path.join(self.vis_root, "VG_100K_2", str(ann["image_id"]) + ".jpg")
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

    def process_text(self, ann):
        question = ann["question"]
        answer = ann["answer"]
        conv = get_conv_template("octopus")
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversation = conv.get_prompt()
        return dict(conversation=conversation)

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True,
                             max_length=self.max_seq_length, add_special_tokens=False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        return res

    def collater(self, samples):
        return collate_fn(samples, pad_idx=self.tokenizer.pad_token_id, eos_idx=self.tokenizer.eos_token_id,
                          left_pad=self.tokenizer.padding_side == "left")


from transformers import LlamaTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

    dataset = LRVDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/visual_genome",
        ann_paths=["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/lrv/filter_cap.json"]
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

        # images = batch['image']
        # image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        # print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        # break

    # media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    # endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    # answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    #
    # for batch in train_dataloader:
    #     batch = batch['net_input']
    #
    #     input_ids = batch["input_ids"]
    #
    #     labels = input_ids.clone()
    #     labels[labels == tokenizer.pad_token_id] = 0
    #     labels[:, 0] = 0
    #
    #     for i in range(labels.shape[0]):
    #         # remove loss for any token before <answer> token
    #         label = [0 for j in range(labels.shape[1])]
    #         left, right = 0, 0
    #         while right < labels.shape[1]:
    #             token_left = labels[i][left]
    #             token_right = labels[i][right]
    #             if token_left == answer_token_id and token_right == endofchunk_token_id:
    #                 label[left: right + 1] = labels[i][left: right + 1]
    #                 right = right + 1
    #                 left = right
    #             elif token_left == answer_token_id and token_right.item() == tokenizer.eos_token_id:
    #                 label[left: right + 1] = labels[i][left: right + 1]
    #                 right = right + 1
    #                 left = right
    #             elif token_left == answer_token_id:
    #                 right = right + 1
    #             else:
    #                 left = left + 1
    #                 right = right + 1
    #         labels[i] = torch.LongTensor(label)
    #
    #     labels[labels == answer_token_id] = 0
    #     labels[labels == media_token_id] = 0
    #
    #     images = batch['image']
    #     image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.savefig("/home/yingzi/vlm/examples/llava.png")
    #
    #     print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
    #     print(tokenizer.decode(labels[1][:sum(batch['attention_mask'][1])]))
    #     break
