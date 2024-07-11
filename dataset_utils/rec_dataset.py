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


### modified from shikra

rec_instructions = [
      "Where is \"<expr>\"?",
      "Where is \"<expr>\" in the image?",
      "Where is \"<expr>\"? answer in [x0,y0,x1,y1] format.",
      "Can you point out \"<expr>\" in the image and provide the coordinates of its location?",
      "Help me to locate \"<expr>\" in the image and give me its coordinates, please.",
      "In the given image, could you find and tell me the coordinates of \"<expr>\"?",
      "Guide me to the location of \"<expr>\" within the image by providing its coordinates.",
      "I'd like to know the exact coordinates of \"<expr>\" in the photo.",
      "Would you kindly provide the coordinates of \"<expr>\" located in the picture?",
      "Can you find \"<expr>\" in the image and give me the coordinates of where it is located?",
      "I'm trying to locate \"<expr>\" in the image. Can you determine its coordinates for me?",
      "What are the coordinates of \"<expr>\" in the image?",
      "Can you disclose the position of \"<expr>\" in the photograph by stating its coordinates?",
      "In the image, could you let me know the location of \"<expr>\" in the form of coordinates?",
      "I need the coordinates of \"<expr>\" in the image, can you please assist me with that?",
      "Where in the image is \"<expr>\" located? Provide me with its coordinates, please.",
      "May I have the coordinates of \"<expr>\" in the image?",
      "In the photograph , could you pinpoint the location of \"<expr>\" and tell me its coordinates?",
      "Can you please search and find \"<expr>\" in the image, then let me know its coordinates?",
      "Please, point out the position of \"<expr>\" in the image by giving its coordinates.",
      "What are the exact coordinates of \"<expr>\" in the provided picture?",
      "Detect the location of \"<expr>\" in the image and share the coordinates with me, please.",
      "In the picture , I'd like you to locate \"<expr>\" and provide its coordinates.",
      "Please indicate the location of \"<expr>\" in the photo by giving coordinates.",
      "Find \"<expr>\" in the image and share its coordinates with me.",
      "Could you please help me find the coordinates of \"<expr>\" in the image?",
      "I am looking for the position of \"<expr>\" in the image. Can you provide its coordinates?",
      "In the image , can you locate \"<expr>\" and let me know its coordinates?",
      "I'd appreciate if you could find and tell me the coordinates of \"<expr>\" in the image.",
      "in the image, I need the bounding box coordinates of \"<expr>\".",
      "Point me to the location of \"<expr>\" in the picture by providing its coordinates.",
      "Could you trace \"<expr>\" in the image and tell me its coordinates?",
      "Can you assist me in locating \"<expr>\" in the image, and then provide its coordinates?",
      "I'm curious, what are the coordinates of \"<expr>\" in the photo?",
      "Kindly share the coordinates of \"<expr>\" located in the image.",
      "I would like to find \"<expr>\" in the image. Can you give me its coordinates?",
      "Can you spot \"<expr>\" in the image and disclose its coordinates to me?",
      "Please, reveal the location of \"<expr>\" in the provided photograph as coordinates.",
      "Help me locate and determine the coordinates of \"<expr>\" in the image.",
      "I request the coordinates of \"<expr>\" in the image.",
      "In the given , can you find \"<expr>\" and tell me its coordinates?",
      "I need to know the position of \"<expr>\" in the image as coordinates.",
      "Locate \"<expr>\" in the image and provide its coordinates, please.",
      "Assist me in finding \"<expr>\" in the photo and provide the bounding box coordinates.",
      "In the image, can you guide me to the location of \"<expr>\" by providing coordinates?",
      "I'd like the coordinates of \"<expr>\" as it appears in the image.",
      "What location does \"<expr>\" hold in the picture? Inform me of its coordinates.",
      "Identify the position of \"<expr>\" in the image and share its coordinates.",
      "I'd like to request the coordinates of \"<expr>\" within the photo.",
      "How can I locate \"<expr>\" in the image? Please provide the coordinates.",
      "I am interested in knowing the coordinates of \"<expr>\" in the picture.",
      "Assist me in locating the position of \"<expr>\" in the photograph and its bounding box coordinates.",
      "In the image , I need to find \"<expr>\" and know its coordinates. Can you please help?"
]


def resize_bbox(bbox, width_ratio, height_ratio):
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = x_min / width_ratio, y_min / height_ratio, x_max / width_ratio, y_max / height_ratio
    x_min, y_min, x_max, y_max = round(x_min, 3), round(y_min, 3), round(x_max, 3), round(y_max, 3)
    return x_min, y_min, x_max, y_max


def process_region(region, original_width, original_height):
    region = resize_bbox(region, original_width, original_height)
    region = [str(r) for r in region]
    return "[" + ' '.join(region) + "]"


class RECDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, **kwargs):
        super().__init__(tokenizer, vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        # image_path = os.path.join(self.vis_root, ann["img_path"])
        # image = Image.open(image_path).convert("RGB")
        image = Image.new('RGB', (224, 224), (0, 0, 0))
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
        expression = ann["expression"]
        bbox = process_region(ann["bbox"], ann["width"], ann["height"])
        question = random.choice(rec_instructions).replace("<expr>", expression)
        answer = bbox

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

    dataset = RECDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=
        ["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/shikra/GC_genome196_train.jsonl",
         ]
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
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
        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        break

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
