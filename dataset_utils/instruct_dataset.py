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
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import LlamaTokenizer, CLIPImageProcessor

from dataset_utils.vqa_dataset import VQADataset, VisualPrompter
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import collate_fn
from conversation import get_conv_template
from constants import (
    OUTPUT_REGION_TASK,
    OUTPUT_IMAGE_CODE_TASK,
    NO_IMAGE_AS_INPUT,
    OPTIONS_REGION_TASK,
    META_REGION_TASK,
    MISSING_TASK
)

def read_task_cluster(ann_path, interleaved_mode, retrieval_type, ablation_type):
    interleaved_samples, pair_samples = [], []
    for task_cluster in os.listdir(ann_path):
        task_cluster_path = os.path.join(ann_path, task_cluster)
        for task in os.listdir(task_cluster_path):
            if ablation_type is None:
                task_interleaved_file = os.path.join(task_cluster_path, task, f"train_interleaved_{retrieval_type}.jsonl")
            else:
                task_interleaved_file = os.path.join(task_cluster_path, task, f"train_interleaved_{retrieval_type}_{ablation_type}.jsonl")
            if os.path.exists(task_interleaved_file) and interleaved_mode:
                raw_data = []
                with open(task_interleaved_file, "r") as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        line['task_name'] = task
                        raw_data.append(line)
                interleaved_samples.extend(raw_data)
            else:
                raw_data = []
                task_file = os.path.join(task_cluster_path, task, "train.jsonl")
                with open(task_file, "r") as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        line['task_name'] = task
                        raw_data.append(line)
                pair_samples.extend(raw_data)

    return (pair_samples, interleaved_samples)


class MultimodalDataset(VQADataset):
    def __init__(self,
                 tokenizer,
                 vis_root,
                 ann_paths,
                 add_eos=True,
                 ignore_instruction=True,
                 max_seq_length=1024,
                 max_num_images=6,
                 prompt_template="octopus",
                 mode="train",
                 task_name=None,
                 with_exemplars=False,
                 with_cot=False,
                 no_region_tasks=False,
                 only_vqa_tasks=False,
                 suffix_loss=False,
                 only_interleaved=False,
                 retrieval_type="image",
                 ablation_type=None,
                 interleaved_mode=None,
    ):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images
        self.vis_root = vis_root
        self.prompt_template = prompt_template
        self.mode = mode
        self.with_exemplars = with_exemplars
        self.icl_examples = None
        self.with_cot = with_cot
        self.suffix_loss = suffix_loss
        self.interleaved_mode = interleaved_mode

        self.pair_samples = []
        self.interleaved_samples = []

        ################## load raw data #####################
        for ann_path in ann_paths:
            if os.path.isdir(ann_path):
                item = read_task_cluster(ann_path, interleaved_mode, retrieval_type, ablation_type)
                self.pair_samples.extend(item[0])
                self.interleaved_samples.extend(item[1])
            else:
                with open(ann_path, "r") as f:
                    self.pair_samples.extend([json.loads(line) for line in f.readlines()])


        ################## fliter tasks ########################

        vcot_template_num, gcot_template_num = 0, 0
        self.annotation = []
        self.icl_key_to_id = {}
        for line in self.pair_samples:
            task = line['task_name']
            unique_id = line['unique_id']
            if no_region_tasks:
                if task in OUTPUT_IMAGE_CODE_TASK or task in MISSING_TASK:  continue
                if task in OUTPUT_REGION_TASK: continue
                if task in OPTIONS_REGION_TASK: continue
                if task in META_REGION_TASK: continue
            if only_vqa_tasks:
                if "vqa" not in task.lower(): continue
            if task_name:
                if task not in task_name: continue
            if ("cot" in task or "cot" in unique_id) and not self.with_cot: continue
            if "options" in line.keys() and len(line["options"]) > 11: continue
            if "gcot" in unique_id: gcot_template_num += 1
            if "vcot" in unique_id: vcot_template_num += 1
            self.icl_key_to_id[line['unique_id']] = len(self.annotation)
            self.annotation.append(line)

        ################## add few-shot templates ########################
        self.icl_template_num, self.icl_template_len = 0, 0
        self.text_icl_templates, self.image_icl_templates = 0, 0
        if self.with_exemplars: self.add_icl_templates()

        ################## add interleaved instruction ########################
        if only_interleaved: self.annotation.clear()
        for line in self.interleaved_samples:
            line['unique_id'] = line['unique_id'] + "_interleaved"
            self.annotation.append(line)

        print(f"total examples num: {len(self.annotation)}")
        print(f"pair examples num: {len(self.pair_samples)}")
        print(f"interleaved examples num: {len(self.interleaved_samples)}")
        print(f"total vcot templates num: {vcot_template_num}")
        print(f"total gcot templates num: {gcot_template_num}")
        print(f"total icl templates num: {self.icl_template_num}")
        print(f"total image icl templates num: {self.image_icl_templates}")
        print(f"total text icl templates num: {self.text_icl_templates}")
        if self.icl_template_num > 0:
            print(f"icl templates avg length: { self.icl_template_len / self.icl_template_num}")

        self.vis_processor = DefaultTransform()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def add_icl_templates(self):
        for i, ann in enumerate(self.annotation):
            icl_examples = self.construct_icl_examples(ann)
            if len(icl_examples) == 0: continue
            icl_content = get_conv_template(self.prompt_template)
            for line in icl_examples:
                icl_content.append_message(icl_content.roles[0], line['prompt'])
                if self.with_cot and "cot_target" in line.keys(): icl_content.append_message(icl_content.roles[1], line['cot_target'])
                else: icl_content.append_message(icl_content.roles[1], line['target'])
            self.annotation[i]["icl_images"] = [line['image_path'] for line in icl_examples]
            icl_content = icl_content.get_prompt()
            self.annotation[i]["icl_examples"] = icl_content
            self.icl_template_num += 1
            self.icl_template_len += len(icl_examples)

    def construct_icl_examples(self, ann):
        image_threshold, text_threshold = 0.8, 0.9
        image_ref_ids, text_ref_ids = {}, {}
        if 'image_ref_ids' in ann.keys():
            image_ref_ids = ann['image_ref_ids']
        if 'text_ref_ids' in ann.keys():
            text_ref_ids = ann['text_ref_ids']

        is_cot_example = False
        if "cot_target" in ann.keys(): is_cot_example = True

        image_icl_examples, text_icl_examples = [], []
        if len(image_ref_ids.keys()) > 0:
            for key, sim in image_ref_ids.items():
                if key not in self.icl_key_to_id.keys(): continue
                if sim >= image_threshold:
                    if is_cot_example:
                        if "cot_target" in self.annotation[self.icl_key_to_id[key]].keys():
                            image_icl_examples.append((sim - image_threshold, self.annotation[self.icl_key_to_id[key]]))
                    else:
                        if "cot_target" not in self.annotation[self.icl_key_to_id[key]].keys():
                            image_icl_examples.append((sim - image_threshold, self.annotation[self.icl_key_to_id[key]]))

        if len(text_ref_ids.keys()) > 0:
            for key, sim in text_ref_ids.items():
                if key not in self.icl_key_to_id.keys(): continue
                if sim >= text_threshold:
                    if is_cot_example:
                        if "cot_target" in self.annotation[self.icl_key_to_id[key]].keys():
                            text_icl_examples.append((sim - image_threshold, self.annotation[self.icl_key_to_id[key]]))
                    else:
                        if "cot_target" not in self.annotation[self.icl_key_to_id[key]].keys():
                            text_icl_examples.append((sim - text_threshold, self.annotation[self.icl_key_to_id[key]]))

        def taskone(elem):
            return elem[0]

        image_icl_examples.sort(key=taskone, reverse=True)
        text_icl_examples.sort(key=taskone, reverse=True)

        icl_examples = []
        if len(image_icl_examples) > 0 and len(text_icl_examples) > 0:
            icl_examples = [item[1] for i, item in enumerate(image_icl_examples)]
            icl_examples.extend([item[1] for i, item in enumerate(text_icl_examples)])
            self.text_icl_templates += 1
            self.image_icl_templates += 1

        elif len(image_icl_examples) > 0:
            icl_examples = [item[1] for i, item in enumerate(image_icl_examples)]
            self.image_icl_templates += 1
        elif len(text_icl_examples) > 0:
            icl_examples = [item[1] for i, item in enumerate(text_icl_examples)]
            self.text_icl_templates += 1
        else:
            icl_examples = []

        return icl_examples

    def process_image(self, ann):
        image_list = []
        if type(ann['image_path']) is list:
            if ann['image_path'][0] == None:
                image_list.extend([Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(len(ann['image_path']))])
            else:
                image_list.extend(Image.open(os.path.join(
                    self.vis_root, self.mode, image_path.split("/")[-1].strip(" "))).convert("RGB") \
                                  for image_path in ann['image_path'])
        elif type(ann['image_path']) is str:
            if "icl_images" in ann.keys() and self.with_exemplars:
                image_list.extend(Image.open(os.path.join(
                    self.vis_root, self.mode, image_path.split("/")[-1].strip(" "))).convert("RGB") \
                                  for image_path in ann['icl_images'])
            image = Image.open(os.path.join(
                 self.vis_root, self.mode, ann["image_path"].split("/")[-1].strip(" "))).convert("RGB")
            image_list.append(image)
        else:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            image_list.append(image)

        self.image_size = image_list[0].size
        images = [self.vis_processor(image) for image in image_list]
        images = torch.stack(images, dim=0)
        keep_ixs = range(min(len(images), self.max_num_images))
        images_tensors = images[keep_ixs]
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        images = images_tensors.unsqueeze(1)
        return images

    def process_text(self, ann):
        if self.interleaved_mode and "interleaved_template" in ann.keys():
            conversation = ann['interleaved_template'][self.interleaved_mode]
            if ann['task_name'] in ['dolly', 'alpaca', 'codex_math', 'flan_mini_cot', 'camel_math', 'share_gpt']:
                conversation = conversation.replace("<image>", "")
            if self.suffix_loss and "<answer>" in conversation:
                conversation = conversation.replace("<answer>", "")
                conversation = conversation.replace("GPT:", "GPT:<answer>")
        else:
            if not isinstance(ann['prompt'], str): ann['prompt'] = str(ann['prompt'])
            if not isinstance(ann['target'], str): ann['target'] = str(ann['target'])

            ### add dialogue history
            dialog_history = None
            if "dialog_hist" in ann.keys():
                dialog_history = get_conv_template(self.prompt_template)
                dialog_history.image_mark = ""
                dialog_history.answer_mark = ""
                for conversations in ann['dialog_hist']:
                    question = conversations["q"]
                    answer = conversations["a"]
                    dialog_history.append_message(dialog_history.roles[0], question)
                    dialog_history.append_message(dialog_history.roles[1], answer)
                dialog_history = dialog_history.get_prompt()

            conv = get_conv_template(self.prompt_template)
            conv.append_message(conv.roles[0], ann["prompt"])

            ### construct cot template
            if self.mode == "train":
                if self.with_cot and "cot_target" in ann.keys():
                    conv.append_message(conv.roles[1], ann["cot_target"])
                else: conv.append_message(conv.roles[1], ann["target"])
            else:
                conv.append_message(conv.roles[1], None)

            ### construct icl template (suffix loss)
            if self.with_exemplars and "icl_examples" in ann.keys():
                conversations = conv.get_prompt()
                if self.suffix_loss: conversation = ann["icl_examples"] + conversations
                else: conversation = ann["icl_examples"].replace("<answer>", "") + conversations
            elif dialog_history:
                conversation = dialog_history + conv.get_prompt()
                conversation = conversation.replace("<image>", "")
                conversation = conversation[:conversation.index("USER:")] + "<image>" + conversation[conversation.index("USER:"):]
            elif ann['image_path'] is None:
                conversation = conv.get_prompt().replace("<image>", "")
            else:
                conversation = conv.get_prompt()

        return dict(conversation=conversation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        task_name, unique_id = ann['task_name'], ann['unique_id']
        image_id = None
        image = self.process_image(ann)  ### speed bottleneck
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(image=image)
        res.update(text)
        res.update(task_name=task_name)
        res.update(image_id=image_id)
        res.update(unique_id=unique_id)
        return res

    def tokenize(self, text):

        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True,
                             max_length=self.max_seq_length, add_special_tokens=False)
        if self.mode == "train":
            res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
            res["attention_mask"] = torch.cat(
                [self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        # else:
        #     res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0)]).unsqueeze(0)
        #     res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0)]).unsqueeze(0)
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

    dataset = MultimodalDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=2000,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="octopus",
        retrieval_type="text",
        ablation_type=None,
    )

    # task_name = [
    #     "open-domain_VQA",
    #     "description_cot_bbox",
    #     "shikra_boxcot",
    #     "llava",
    #     "llava_dial",
    # ],

    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
        shuffle=True,
    )

    from tqdm import tqdm

    pbar = tqdm(total=len(dataset))
    cnt = 0

    from collections import defaultdict


    def zero(): return 0
    task_num = defaultdict(zero)

    # task_sent_len = defaultdict(list)
    #
    # for batch in train_dataloader:
    #     task = batch['task_name'][0]
    #     task_sent_len[task].append(batch['net_input']["input_ids"].shape[1])
    #     pbar.update(1)
    #
    # for task in task_sent_len.keys():
    #     lens = task_sent_len[task]
    #     print(f"========{task}=========")
    #     print(max(lens))
    #     print(min(lens))
    #     print(sum(lens) / len(lens))
    #
    #     task_sent_len[task] = sum(lens) / len(lens)
    #
    # print(task_sent_len)

    cnt = 0
    for batch in train_dataloader:
        pbar.update(1)
        task = batch['task_name'][0]
        batch = batch['net_input']

        if "caption" not in task: continue

        # for k, v in batch.items():
        #     if isinstance(v, list):
        #         print(k, v)
        #         continue
        #     print(k, v.shape)

        images = batch['image']
        # image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        #
        # # plt.imshow(image)
        # # plt.savefig(f"/home/scratch.chaoweix_nvresearch/visual_instruction/Visual-Instrcution-tuning/examples/flan/{task}.png")
        #
        # start_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
        # media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        # endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        # answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
        #
        # input_ids = batch["input_ids"]
        # labels = input_ids.clone()
        # labels[labels == tokenizer.pad_token_id] = 0
        # labels[:, 0] = 0
        #
        # for i in range(labels.shape[0]):
        #     endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
        #     media_idxs = torch.where(labels[i] == media_token_id)[0]
        #
        #     # remove loss for any token the before the first <answer>
        #     token_idx = 0
        #     while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
        #         labels[i][token_idx] = 0
        #         token_idx += 1
        #
        #     # # remove loss for any token between <|endofchunk|> and <answer>, except <image>
        #     # for endofchunk_idx in endofchunk_idxs:
        #     #     token_idx = endofchunk_idx + 1
        #     #     while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
        #     #         if labels[i][token_idx] == media_token_id:
        #     #             pass
        #     #         else:
        #     #             labels[i][token_idx] = -100
        #     #         token_idx += 1
        #
        # labels[labels == answer_token_id] = 0
        # labels[labels == media_token_id] = 0

        task_num[task] += 1
        if task_num[task] == 1:
            cnt += 1
            print("=" * 25 + str(cnt) + "=" * 25)
            print(task)
            print(images.shape)
            print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
            # print(tokenizer.decode(labels[0][:sum(batch['attention_mask'][0])]))

