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

reg_instructions = [
    "For the given image, can you provide a unique description of the area <objs>?",
    "Please generate a distinguishing description for the region <objs> in the image.",
    "In the photo, how would you describe the selected area <objs> uniquely?",
    "Given the image, I'd like you to describe the area <objs> in a distinguishing manner.",
    "Can you provide a description for the region <objs> in the image such that it sets it apart from others?",
    "In the image, I want a unique description of the specified region <objs>.",
    "In the image, describe the selected area <objs>, making sure your description is unique.",
    "Could you describe the distinct features of the area <objs> in the picture?",
    "Help me understand what makes the area <objs> unique in the image.",
    "Please describe the area <objs> in the image in a way that differentiates it from others.",
    "How can you uniquely describe the selected area <objs> found in the image?",
    "Generate a distinct description of the region <objs> with in the image.",
    "In the provided image, can you describe the area <objs> in a way that sets it apart?",
    "How would you describe the unique characteristics of the area <objs> in the image?",
    "Can you give me a description of the area <objs> in the image that clearly distinguishes it?",
    "Create a distinguishing description for the selected area <objs> in the picture.",
    "What is a unique description for the area <objs> in the image?",
    "Describe the area <objs> in a unique way, given the picture.",
    "Please provide a description that sets the area <objs> apart from others in the image.",
    "How can you characterize the region <objs> in the image to make it stand out?",
    "I'd like a description of the area <objs> that differentiates it from other regions in the image.",
    "For the specified area <objs> in the photo, please provide a distinctive description.",
    "In the image, can you give me a unique description of the selected area <objs>?",
    "How would you describe the special characteristics of the region <objs> in the image?",
    "What makes the area <objs> stand out in the image? Please provide a detailed description.",
    "Please generate a unique description for the area <objs> displayed in the image.",
    "Create a one-of-a-kind description for the region <objs> found in the picture.",
    "Describe the distinct aspects of the area <objs> present in the photo.",
    "Help me identify the unique characteristics of the region <objs> in the image.",
    "Can you provide a description of the selected area <objs> in the image that sets it apart?",
    "How would you describe the specific region <objs> in the given image?",
    "I'd like an exclusive description for the area <objs> in the photograph.",
    "Generate a description that highlights the uniqueness of the area <objs> in the image.",
    "Can you describe the area <objs> in such a way that it differs from other regions in the image?",
    "I'm looking for a detailed description of the unique features with in the area <objs> in the image.",
    "Please describe the selected region <objs> in the image in a distinctive manner.",
    "What can you tell me about the area <objs> in the image that sets it apart from the rest?",
    "Describe the region <objs> in the image, focusing on its unique attributes.",
    "Create a description of the area <objs> in the photo that emphasizes its distinctiveness.",
    "Generate a one-of-a-kind description of the specified region <objs> in the image.",
    "Can you describe the distinct qualities of the area <objs> in the provided picture?",
    "In the image, provide a unique description of the selected region <objs>.",
    "Describe what makes the area <objs> special in the context of the image.",
    "Provide a description that sets the region <objs> apart from others in the picture.",
    "Can you give me a distinguishing description of the area <objs> in the photo?",
    "How can you uniquely characterize the selected region <objs> in the image?",
    "Can you provide a unique description for the designated region <objs> with in the image?",
    "What features distinguish the specified bounding box <objs> in the image from other areas?",
    "Describe the unique characteristics of the area <objs> in the image.",
    "In the image, how would you distinctly describe the selected area <objs>?",
    "Please elaborate on the specific attributes of the rectangular region <objs> in the image.",
    "What sets apart the chosen bounding box <objs> in the image from the rest of the image?",
    "Identify and describe the unique features of the bounded area <objs> in the image.",
    "Please provide a one-of-a-kind description for the area marked by <objs> in the image.",
    "How would you distinctly characterize the selected region <objs> in the image?",
    "Tell me about the unique aspects of the region <objs> in the image.",
    "Highlight the distinctive characteristics of the bounding box <objs> in the image.",
    "Provide a detailed description that sets apart the area <objs> in the image from its surroundings.",
    "In the image, describe the highlighted region <objs> in a way that distinguishes it from other areas.",
    "What attributes make the area enclosed by <objs> in the image unique compared to the rest of the image?",
    "Focus on the distinct elements of the selected bounding box <objs> in the image and describe them.",
    "Give me a singular description for the specified region <objs> in the picture.",
    "Describe the exclusive features of the rectangular box <objs> depicted in the image.",
    "Please point out and discuss the uniqueness of the chosen area <objs> in the image.",
    "What are the distinguishing factors of the marked section <objs> in the image?",
    "Kindly provide a distinctive explanation of the area <objs> with in the image.",
    "Offer a unique description of the rectangular region <objs> in the image.",
    "Please focus on and describe the special attributes of the selected box <objs> in the image.",
    "What sets the designated bounding box <objs> in the image apart from its surroundings?",
    "How does the area <objs> in the image stand out uniquely from the rest?",
    "Describe the standout features of the specified bounding box <objs> in the image.",
    "Please explain the exceptional characteristics of the marked area <objs> in the image.",
    "In the image, provide a distinguishing description for the bounded region <objs>.",
    "What are the unique aspects of the selected rectangular area <objs> in the image?",
    "How would you differentiate the area <objs> in the image from others in a description?",
    "Please point out the remarkable features of the specified area <objs> in the image.",
    "Describe the peculiar characteristics of the region enclosed by <objs> in the image.",
    "What are the distinguishing details of the selected box <objs> in the image?",
    "In your own words, how does the specified region <objs> in the image stand out?",
    "Provide a distinguishing summary of the rectangular section <objs> in the image.",
    "What makes the chosen bounding box <objs> in the image different from the rest?",
    "Elaborate on the distinctive properties of the area marked by <objs> in the image.",
    "How can the designated area <objs> in the image be uniquely described?",
    "Please offer a one-of-a-kind explanation for the marked section <objs> in the image.",
    "Discuss the outstanding characteristics of the bounding box <objs> in the image.",
    "Could you analyze the distinct features of the selected region <objs> in the image?",
    "In the image, provide a unique depiction of the specified area <objs>.",
    "What are the notable attributes of the area enclosed by <objs> in the image?",
    "Please describe the exceptional aspects of the rectangular region <objs> in the image.",
    "How would you uniquely characterize the bounding box <objs> in the image?",
    "Focus on differentiating factors when describing the marked area <objs> in the image.",
    "Provide a detailed account of the unique elements with in the region <objs> in the image.",
    "Identify and discuss the exceptional features of the selected box <objs> in the image.",
    "In your own words, what sets the designated area <objs> in the image apart?",
    "How can the specified bounding box <objs> in the image be distinctly described?",
    "What unique features can you point out about the area <objs> in the image?",
    "Please offer a differentiating explanation for the region enclosed by <objs> in the image.",
    "How would you describe the peculiar characteristics of the selected box <objs> in the image?",
    "Give a standout description of the specified area <objs> located in image.",
    "Identify and elaborate on the distinguishing aspects of the rectangular region <objs> in the image.",
    "What makes the bounded area <objs> in the image dissimilar from the rest?",
    "Please provide an inimitable description of the marked section <objs> in the image.",
    "In what way does the designated bounding box <objs> in the image differ from other areas?",
    "How can the unique features of the selected area <objs> in the image be summarized?",
    "Please discuss the exceptional traits of the bounding box <objs> with in the image.",
    "Offer a distinct analysis of the area defined by <objs> in the image.",
    "How does the specified region <objs> in the image differ from its surroundings?",
    "Describe the standout aspects of the rectangular box <objs> in the image.",
    "What makes the chosen area <objs> unique in the context of?",
    "Please provide a dissimilar description of the selected region <objs> in the image.",
    "In what ways is the enclosed area <objs> in the image different from others?",
    "How would you describe the unique elements of the bounding box <objs> in the image?",
    "Please focus on the distinct qualities of the marked area <objs> with in the image.",
    "What sets the specified rectangular region <objs> in image apart from the rest?",
    "Offer a unique perspective on the designated section <objs> in the image.",
    "In the image, discuss the one-of-a-kind attributes of the area <objs>.",
    "What outstanding features does the selected box <objs> in the image possess?",
    "Please describe the exceptional aspects of the defined region <objs> in the image.",
    "How would you distinctly portray the specified area <objs> in the image?",
    "What are the unique characteristics of the rectangular section <objs> in the image?",
    "Describe the novel qualities of the selected bounding box <objs> in the image.",
    "What sets the chosen region <objs> in the image apart from its surroundings?",
    "Provide a one-of-a-kind depiction for the area enclosed by <objs> in the image.",
    "How would you portray the unique features of the designated box <objs> in the image?",
    "Explain the distinguishing characteristics of the marked bounding box <objs> in the image."
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


class REGDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, **kwargs):
        super().__init__(tokenizer, vis_root, ann_paths, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["img_path"])
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
        expression = ann["expression"]
        bbox = process_region(ann["bbox"], ann["width"], ann["height"])
        question = random.choice(reg_instructions).replace("<objs>", bbox)
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

    dataset = REGDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=
        ["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/shikra/REC_ref3_train.jsonl",
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
