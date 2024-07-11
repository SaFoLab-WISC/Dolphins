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

gc_instructions = [
  "Can you give me a description of the region <objs> in image?",
  "In the provided image, would you mind describing the selected area <objs>?",
  "I need details about the area <objs> located within image.",
  "Could you please share some information on the region <objs> in this photograph?",
  "Describe what's happening within the coordinates <objs> of the given image.",
  "What can you tell me about the selected region <objs> in the photo?",
  "Please, can you help me understand what's inside the region <objs> in image?",
  "Give me a comprehensive description of the specified area <objs> in the picture.",
  "I'm curious about the area <objs> in the following image. Can you describe it?",
  "Please elaborate on the area with the coordinates <objs> in the visual.",
  "In the displayed image, help me understand the region defined by <objs>.",
  "Regarding the image, what's going on in the section <objs>?",
  "In the given photograph, can you explain the area with coordinates <objs>?",
  "Kindly describe what I should be seeing in the area <objs> of image.",
  "Within the input image, what can be found in the region defined by <objs>?",
  "Tell me what you see within the designated area <objs> in the picture.",
  "Please detail the contents of the chosen region <objs> in the visual input.",
  "What's inside the area <objs> of the provided graphic?",
  "I'd like some information about the specific region <objs> in the image.",
  "Help me understand the details within the area <objs> in photograph.",
  "Can you break down the region <objs> in the image for me?",
  "What is taking place within the specified area <objs> in this capture?",
  "Care to elaborate on the targeted area <objs> in the visual illustration?",
  "What insights can you provide about the area <objs> in the selected picture?",
  "What does the area <objs> within the given visual contain?",
  "Analyze and describe the region <objs> in the included photo.",
  "Please provide details for the area marked as <objs> in this photographic.",
  "For the image, can you assess and describe what's happening at <objs>?",
  "Fill me in about the selected portion <objs> within the presented image.",
  "In the image, elaborate on the details found within the section <objs>.",
  "Please interpret and describe the area <objs> inside the given picture.",
  "What information can you give me about the coordinates <objs> in image?",
  "Regarding the coordinates <objs> in image, can you provide a description?",
  "In the photo, can you delve into the details of the region <objs>?",
  "Please provide insights on the specified area <objs> within the graphic.",
  "Detail the chosen region <objs> in the depicted scene.",
  "Can you discuss the entities within the region <objs> of image?",
  "I'd appreciate a breakdown of the area <objs> in the displayed image.",
  "What's the story in the section <objs> of the included visual?",
  "Please enlighten me about the region <objs> in the given photo.",
  "Offer a thorough description of the area <objs> within the illustration.",
  "What can you share about the area <objs> in the presented image?",
  "Help me grasp the context of the region <objs> within image.",
  "Kindly give an overview of the section <objs> in photo.",
  "What details can you provide about the region <objs> in the snapshot?",
  "Can you divulge the contents of the area <objs> within the given image?",
  "In the submitted image, please give a synopsis of the area <objs>.",
  "In the image, please describe the bounding box <objs>.",
  "Please describe the region <objs> in the picture.",
  "Describe the bbox <objs> in the provided photo.",
  "What can you tell me about the area <objs> within the image?",
  "Could you give me a description of the rectangular region <objs> found in the image?",
  "in the image, what elements can be found within the coordinates <objs>?",
  "Please provide details for the area within the bounding box <objs> in the image.",
  "Can you generate a description for the selected region <objs> in the image?",
  "Kindly describe the objects or scenery in the bounding box <objs> within the image.",
  "What details can you provide for the rectangle defined by the coordinates <objs> in the image?",
  "In relation to the picture, please describe the content of the area marked by <objs>.",
  "I'd like to know more about the area <objs> in the given image. Can you describe it?",
  "Can you help me by describing the part of that lies within the bounding box <objs>?",
  "What's happening in the section of the photo enclosed by the coordinates <objs>?",
  "Describe the image content present in the specified rectangular area <objs> of.",
  "Please provide information about the area within the bounding box <objs> in the picture.",
  "Could you offer a description of the contents in the selected area <objs> of the image?",
  "I'm curious about the area <objs> in the image. Can you provide a description of it?",
  "What can be observed in the rectangular region <objs> in the photograph?",
  "Please explain what is contained in the portion of defined by the box <objs>.",
  "In the photograph, can you describe the objects or scenery enclosed by <objs>?",
  "Can you give a brief explanation of the specified area <objs> in the image?",
  "What does the area <objs> look like in the context of the image?",
  "Could you please describe the contents of the bounding box <objs> in the given image?",
  "I would like to know more about the rectangular region <objs> within the picture. Can you describe it?",
  "Please tell me about the area <objs> in the image. What does it contain?",
  "Help me understand what's happening in the selected bounding box <objs> within the image.",
  "Can you provide a description of the area <objs> in the image?",
  "What sort of things can be seen in the region <objs> of the photo?",
  "Describe what can be found within the bounds of <objs> in the image.",
  "in the image, can you paint a picture of the area enclosed by coordinates <objs>?",
  "Please provide a detailed account of the area covered by the bounding box <objs> in the image.",
  "Give me a vivid description of what's happening in the area <objs> within the snapshot.",
  "In the image, what do you observe within the rectangular box defined by the coordinates <objs>?",
  "Could you give me a breakdown of the content in the specified area <objs> of the picture?",
  "Please elucidate the area<objs> of the image.",
  "I'd appreciate it if you could describe the portion of that lies within the rectangle <objs>.",
  "Can you share some insights about the rectangular region <objs> in the image?",
  "Help me visualize the section of the photo enclosed by the bounding box <objs>.",
  "Would you kindly provide a description for the content within the rectangular area <objs> of?",
  "in the image, can you tell me more about the area specified by the bounding box <objs>?",
  "Please describe what can be seen in the rectangular region <objs> of the image.",
  "Can you analyze the content of the area <objs> within the photograph?",
  "In the provided image, please explain the content within the region <objs>.",
  "I'm interested in the selected rectangle <objs> in the image. Can you tell me more about it?",
  "Explain what can be found in the bounding box <objs> in the context of the image.",
  "Kindly share your observations about the rectangular region <objs> within the image.",
  "I'd like a thorough description of the area <objs> in the image.",
  "Could you please provide a description of the rectangular area <objs> in the image?",
  "Please describe the section of the picture defined by the bbox <objs>.",
  "Tell me more about the scenery or objects within the rectangular region <objs> in the image.",
  "Would you kindly describe the content of the area enclosed by <objs> in the image?",
  "Help me understand the objects or scenery within the bounding box <objs> in the image.",
  "I would like to know about the section of the image enclosed by the rectangle <objs>. Can you describe it?",
  "Describe the selected rectangular area <objs> in the photo.",
  "Tell me about the region <objs> of the image.",
  "I request a description of the area <objs> in the picture.",
  "Can you elaborate on the content of the bounding box <objs> in the image?",
  "Please share details about the rectangular region <objs> within the image.",
  "What can I find in the bbox <objs> of the provided image?",
  "In the image, could you provide a description for the coordinates <objs>?",
  "Could you tell me more about the area <objs> in the snapshot?",
  "Fill me in on the details of the rectangular box <objs> within the image.",
  "What's going on in the section of contained within the bounding box <objs>?",
  "I would like a description of the content within the bbox <objs> in the image.",
  "Please enlighten me about the area <objs> in the photograph.",
  "Can you give me a visual rundown of the area <objs> in the image?",
  "Describe the visual elements within the selected area <objs> of the image.",
  "Tell me what you see in the area <objs> within the context of the image.",
  "Explain the content within the rectangular region <objs> of the image.",
  "I'd like some information about the bounding box <objs> in the photo.",
  "What is happening within the rectangle defined by coordinates <objs> in the image?",
  "Please describe the content within the area <objs> displayed in the image.",
  "What can be seen in the bounding box <objs> in the context of the provided image?",
  "Share some details about the objects or environment within the bounding box <objs> in the image.",
  "Please describe the area <objs> in the image for me.",
  "Can you generate a description of the contents within the selected region <objs> in the image?",
  "What objects or scenery can be found in the area <objs> in the image?",
  "Please tell me more about the rectangular section <objs> in the photo.",
  "Could you describe the content of the bbox <objs> in the image?",
  "What does the selected region <objs> in the image encompass?",
  "I am interested in the region <objs> of the image; please describe it.",
  "Can you provide some context for the area <objs> within the picture?",
  "Please give me some details about the rectangle <objs> in the image.",
  "In the photo, what can you see within the region defined by the bounding box <objs>?",
  "I would like a detailed description of the portion of enclosed by the bbox <objs>.",
  "Please help me understand the content present within the rectangle <objs> in the image.",
  "Would you mind describing the rectangular area <objs> in the provided image?"
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


class GCDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, **kwargs):
        super().__init__(tokenizer, vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        # image_path = self.vis_root + ann["img_path"][ann["img_path"].index("/"):]
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
        question = random.choice(gc_instructions).replace("<objs>", bbox)
        answer = expression

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

    dataset = GCDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/visual_genome",
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
