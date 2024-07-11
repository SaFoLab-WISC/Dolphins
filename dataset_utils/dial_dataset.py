import sys

sys.path.append("..")

from dataset_utils.vqa_dataset import VQADataset
from dataset_utils.processors import DefaultTransform
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
from conversation import get_conv_template

class DialDataset(VQADataset):
    def __init__(self, *args, **kwargs):
        super(DialDataset, self).__init__(*args, **kwargs)

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, "COCO_train2014_" + ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = [self.vis_processor(image)]
        region_images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]
        # pad to 5 images
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        image = images_tensors.unsqueeze(1)
        return image

    def process_text(self, anns):
        # TODO remove this
        dialog_history = get_conv_template("octopus")
        dialog_history.image_mark = ""
        dialog_history.answer_mark = ""
        for conv_id in range(0, len(anns["conversations"]) - 2, 2):
            question = anns["conversations"][conv_id]["value"]
            question = question.replace("<image>", "").replace("\n", "")
            answer = anns["conversations"][conv_id + 1]["value"]
            dialog_history.append_message(dialog_history.roles[0], question)
            dialog_history.append_message(dialog_history.roles[1], answer)
            
        utterance = get_conv_template("open_flamingo_v0")
        utterance.system = ""
        utterance.image_mark = ""
        for conv_id in range(len(anns["conversations"]) - 2, len(anns["conversations"]), 2):
            question = anns["conversations"][conv_id]["value"]
            question = question.replace("<image>", "").replace("\n", "")
            answer = anns["conversations"][conv_id + 1]["value"]
            utterance.append_message(utterance.roles[0], question)
            utterance.append_message(utterance.roles[1], answer)

        conversation = dialog_history.get_prompt() + utterance.get_prompt()
        conversation = conversation[:conversation.index("USER:")] + \
                       "<image>" + \
                       conversation[conversation.index("USER:"):]

        return dict(conversation=conversation)

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True,max_length=self.max_seq_length, add_special_tokens=False)
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

    image_processor = DefaultTransform()

    dataset = DialDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=
        ["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/llava/conversation_58k.json"]
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        #sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )
    
    from tqdm import tqdm

    pbar = tqdm(total=len(train_dataloader))
    for batch in train_dataloader:
        batch = batch['net_input']
        pbar.update(1)

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
        images = batch['image']
        image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        # plt.imshow(image)
        # plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/Visual-Instrcution-tuning/examples/dial.png")

        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
        #print(tokenizer.decode(labels[0][:sum(batch['attention_mask'][0])]))
        break
