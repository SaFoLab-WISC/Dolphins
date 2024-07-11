"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

from dataset_utils.processors import DefaultTransform

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import LlamaTokenizer

TEMPLATE = {
    "system": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
    "prompt_conv_with_image": ["Human: {image}", "Human: {question} AI: <answer> {response} <|endofchunk|>"],
}

class VQAPrompter:
    def __call__(self, question, response, num_images=1, num_convs=1):
        res = TEMPLATE["system"]
        for i in range(num_images): res += TEMPLATE["prompt_conv_with_image"][0].format(image="<image>")
        for i in range(num_convs): res += TEMPLATE["prompt_conv_with_image"][1].format(question=question[i], response=response[i])
        return res

class VisualPrompter:
    def __call__(self, question, response, num_images=1, num_convs=1):
        res = TEMPLATE["system"]
        for i in range(num_images): res += TEMPLATE["prompt_conv_with_image"][0].format(image="<image>")
        for i in range(num_convs): res += TEMPLATE["prompt_conv_with_image"][1].format(question=question[i], response=response[i])
        return res

class VQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        vis_root=None,
        ann_paths=[],
        add_eos=True,
        ignore_instruction=True,
        sample_image=False,
        max_seq_length=512,
        max_num_images=5,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            try:
                self.annotation.extend(json.load(open(ann_path, "r")))
            except:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f.readlines()])

        self.sample_image = sample_image
        if self.sample_image:
            print("randomly sample one annotation for each image")
            self.annotation = self.parse_annotation(self.annotation)

        self.vis_processor = DefaultTransform()
        self.prompter = VisualPrompter()

        self._add_instance_ids()
        self.option_prob = 0.5
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def parse_annotation(self, annotation):
        image_list = defaultdict(list)
        for ann in annotation:
            image_list[ann["image"]].append(ann)
        # image_name_list = list(image_list.keys())
        annotation = []
        for ann_list in image_list.values():
            annotation.append(random.choice(ann_list))
        return annotation

    def __len__(self):
        return len(self.annotation)

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return image

    def process_text(self, ann):
        question = ann["question"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        # create instruction
        true_answer = answers[np.argmax(weights)]
        is_option = random.random() < self.option_prob and len(answers) > 1
        if is_option:
            instruction = self.prompter(question, answers)
        else:
            instruction = self.prompter(question)

        return dict(instruction=instruction, answer=true_answer)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if res["input_ids"][-1] != self.tokenizer.eos_token_id and len(res["input_ids"]) < 512 and self.add_eos:
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [-100] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = self.process_image(ann)
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(image=image)
        res.update(text)
        return res

    def collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
