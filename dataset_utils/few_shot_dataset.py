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

def build_interleaved_instruction(task,
                                  text=None,
                                  options=None,
                                  region=None,
                                  context=None,
                                  question=None,
                                  explanation=None,
                                  response=None,
                                  premise=None,
                                  hypothesis=None,
                                  answer=None,
                                  meta_data=None,
                                  prompt=None,
                                  target=None,
                                  use_natural=False,
                                  instruction_id=-1,
                                  icl_example_id=0,
                                  is_icl_example=False,
                                  mode="train"):

    if task == "image_caption_flickr30k":
        general_instructs = [
            f"""In this task, you will look at following images and generate a long sentence to describe each image.""",
            f"""What is the caption of the given images? Give a long sentence for each image.""",
            f"""Generate a text to describe each image.""",
            f"""Look at image and tell me what is the content. Output a short sentence for each image.""",
            f"""In this task, you are given some images and you will need to generate some text to describe them respectively."""
        ]

        instructs = [
            f"""Generate a long sentence to describe this image {icl_example_id}: <target> """,
            f"""A short description of this image {icl_example_id} is: <target> """,
            f"""The image {icl_example_id} shows a photo of <target> """,
            f"""The image {icl_example_id} depicts a scene of <target> """,
            f"""The image {icl_example_id} showcases a sence of <target> """,
            f"""When we look at this picture, we can see: <target> """,
            f"""The image {icl_example_id} depicts: <target> """,
            f"""This image conveys to us the content of <target> """,
            f"""Upon observing the details within the image {icl_example_id}, one can conclude that: <target> """,
        ]

    elif task == 'ok_vqa':
        general_instructs = [
            f"""In this task, you will be asked some questions about the following images. However, in order to answer these questions, you need knoweldge outside of images.""",
            f"""Use external knoweldge and the content of the image to briefly answer the question.""",
            f"""Based on the content of the given images and external knowledge, briefly answer the corresponding questions.""",
            f"""Based on your knowledge, output a short answer for each question about the following images.""",
            f"""Given some images and questions, use your external knowledge and common sense to briefly answer these questions.""",
        ]

        instructs = [
            f"""Question about image {icl_example_id}: {question} Answer: <target>""",
            f"""The question about this image: {question} Answer: <target>""",
            f"""Given the image {icl_example_id} and answer the question by using your external knowledge: {question} Answer: <target>""",
            f"""Question about this image: {question} Answer: <target>""",
            f"""Use image {icl_example_id} and your external knowledge to answer the question: {question} Answer: <target>""",
        ]

    elif "vizwiz" in task:
        general_instructs = [
            f"""In this task, you will be asked some questions about the following images. Determine whether the question can be answered, if so, then give the answer to the question, otherwise output \"unanswerable\". """,
            f"""In this task, you will be asked some questions about the following images. """,
            f"""In this task, you will be asked some questions about the following images. """,
            f"""In this task, you will be asked some questions about the following images. """,
            f"""In this task, you will be asked some questions about the following images. """,
        ]

        instructs = [
            f"""The question about image {icl_example_id} is: {question} Answer:<target>""",
            f""" is image {icl_example_id}. {question} Answer:<target>""",
            f"""The question about this image: {question} Answer:<target>""",
            f"""Given the image {icl_example_id} and answer the question by using your external knowledge: {question} Answer: <target>""",
            f"""Question about this image: {question} Answer: <target>""",
            f"""Use image {icl_example_id} and your external knowledge to answer the question: {question} Answer: <target>""",
        ]
    elif "hateful_meme" in task:
        general_instructs = [
            f"""In this task, you will be asked whether the text in the following meme images is hateful.""",
            f"""In this task, you will be asked whether the text in the following meme images is hateful.""",
            f"""In this task, you will be asked whether the text in the following meme images is hateful.""",
            f"""In this task, you will be asked whether the text in the following meme images is hateful.""",
            f"""In this task, you will be asked whether the text in the following meme images is hateful.""",
        ]

        instructs = [
            f""" is an image with: \"{question}\" written on it. Is it hateful? Answer: <target>""",
            f""" is an image with: \"{question}\" written on it. Is it hateful? Answer: <target>""",
            f""" is an image with: \"{question}\" written on it. Is it hateful? Answer: <target>""",
            f""" is an image with: \"{question}\" written on it. Is it hateful? Answer: <target>""",
            f""" is an image with: \"{question}\" written on it. Is it hateful? Answer: <target>""",
        ]

    else:
        general_instructs = [
            f"""In this task, you will be asked some questions about the following images. However, in order to answer these questions, you need the text on the image.""",
            f"""Use some text on the image to briefly answer the question.""",
            f"""Based on the following images and the text on the image, briefly answer the corresponding questions.""",
            f"""Based on the text on the following images, output a short answer for each question about the image.""",
            f"""Given some images and questions, use the text on the image to briefly answer these questions.""",
        ]

        instructs = [
            f"""Question about image {icl_example_id}: {question} Answer: <target>""",
            f"""The question about this image: {question} Answer: <target>""",
            f"""Given the image {icl_example_id} and answer the question by using your external knowledge: {question} Answer: <target>""",
            f"""Question about this image: {question} Answer: <target>""",
            f"""Use image {icl_example_id} and your external knowledge to answer the question: {question} Answer: <target>""",
        ]

    assert len(general_instructs) >= 5, f"task {task} has only {len(general_instructs)} instructions"
    assert len(instructs) >= 5, f"task {task} has only {len(instructs)} instructions"

    if not isinstance(target, list): target = [target]

    general_inst = general_instructs[0]
    prompt = instructs[0]
    target = target[0]
    return general_inst, prompt, target


class FewShotEvalDataset(VQADataset):
    def __init__(self, tokenizer, vis_root, ann_paths, add_eos=True, ignore_instruction=True,
                 max_seq_length=512, max_num_images=5, num_shots=2, prompt_template="octopus", interleaved_type=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = num_shots + 1
        self.num_shots = num_shots
        self.vis_root = vis_root
        self.prompt_template = prompt_template
        self.interleaved_type =  interleaved_type

        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                data = [json.loads(line) for line in f.readlines()]
            self.annotation.extend(data)

        self.vis_processor = DefaultTransform()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def process_image(self, ann):
        images = []
        try:
            if self.num_shots > 0:
                icl_index = ann['few_shot'][str(self.num_shots) + "-shot"]
                images = [Image.open(self.annotation[idx]['image_path']).convert("RGB") for idx in icl_index]
            images.append(Image.open(ann["image_path"]).convert("RGB"))
        except:
            image_num = self.num_shots + 1
            images.extend([Image.new('RGB', (224, 224), (0, 0, 0)) for _ in range(image_num)])

        images = [self.vis_processor(image) for image in images]
        region_images = torch.stack(images, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]
        image = images_tensors.unsqueeze(1)

        return image

    def clean_prompt(self, prompt):
        prompt_clean = prompt.replace(" <target>", "")
        prompt_clean = prompt_clean.replace("<target>", "")
        prompt_clean = prompt_clean.replace("Answer:", "")
        return prompt_clean

    def process_text(self, ann):
        icl_examples = None
        if self.interleaved_type is None:
            if self.num_shots > 0:
                icl_examples = get_conv_template(self.prompt_template)
                icl_index = ann['few_shot'][str(self.num_shots) + "-shot"]
                for idx in icl_index:
                    icl_examples.append_message(icl_examples.roles[0], self.annotation[idx]['prompt'])
                    icl_examples.append_message(icl_examples.roles[1], self.annotation[idx]['target'])

                icl_examples = icl_examples.get_prompt()

            conv = get_conv_template(self.prompt_template)
            conv.append_message(conv.roles[0], ann["prompt"])
            conv.append_message(conv.roles[1], None)
            conversation = conv.get_prompt()
            if icl_examples:
                conversation = icl_examples.replace("<answer>", "") + conversation
        else:
            in_context_prompts = []
            if self.num_shots > 0:
                icl_index = ann['few_shot'][str(self.num_shots) + "-shot"]
                in_context_exemplars = [self.annotation[idx] for idx in icl_index]
                for i, icl_exemplar in enumerate(in_context_exemplars):
                    general_inst, prompt, target = build_interleaved_instruction(
                        ann['task_name'],
                        question=icl_exemplar.get('prompt'),
                        target=icl_exemplar.get('target'),
                        icl_example_id=i + 1,
                        is_icl_example=True,
                    )
                    in_context_prompts.append((general_inst, prompt, target))

            general_inst, prompt, target = build_interleaved_instruction(
                ann['task_name'],
                question=ann.get('prompt'),
                target=ann.get('target'),
                icl_example_id=len(in_context_prompts) + 1,
                is_icl_example=False,
            )

            interleaved_data = {}

            text = ""
            for _, icl_prompt, tgt in in_context_prompts:
                text += "<image>" + icl_prompt.replace("<target>", tgt) + "<|endofchunk|>"
            text += "<image>" + prompt.replace("<target>", "")
            assert text.count("<image>") == self.num_shots + 1
            interleaved_data['openflamingo'] = text.strip(" ")

            text = ""
            general_inst = general_inst.replace("images", "image")
            general_inst = general_inst.replace("questions", "question")
            for inst, icl_prompt, tgt in in_context_prompts:
                text += "<image>" + "USER: " + self.clean_prompt(icl_prompt) + "GPT: " + tgt + "<|endofchunk|>"
            text += "<image>" + "USER: " + self.clean_prompt(prompt) + "GPT:<answer>"
            assert text.count("<image>") == self.num_shots + 1
            interleaved_data['otter'] = text.strip(" ")

            text = ""
            text = f"SYSTEM: {general_inst} "
            for _, icl_prompt, tgt in in_context_prompts:
                text += "USER: " + "<image>" + self.clean_prompt(icl_prompt) + "GPT: " + tgt + "<|endofchunk|>"
            text += "USER: " + "<image>" + self.clean_prompt(prompt) + "GPT:<answer>"
            assert text.count("<image>") == self.num_shots + 1
            interleaved_data['octopus'] = text.strip(" ")

            conversation = interleaved_data[self.interleaved_type]

        return dict(conversation=conversation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        task_name, unique_id = ann['task_name'], ann['unique_id']
        if ann['image_path']:
            image_id = ann['image_path'].split("/")[-1].strip(" ").split(".")[0].split("_")[-1].lstrip("0")
        else:
            image_id = None
        image = self.process_image(ann) ### speed bottleneck
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
        # res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        # res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
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

    dataset = FewShotEvalDataset(
        tokenizer=tokenizer,
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=
        ["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/test_few_shot.jsonl",
         ],
        num_shots=8,
        max_seq_length=1024,
        interleaved_type="octopus",
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    from tqdm import tqdm
    from collections import defaultdict
    def zero(): return 0
    task_num = defaultdict(zero)

    pbar = tqdm(total=len(train_dataloader))
    for batch in train_dataloader:
        pbar.update(1)
        task_name = batch['task_name'][0]
        task_num[task_name] += 1
        if "caption" not in task_name: continue
        batch = batch['net_input']
        if task_num[task_name] == 1:
            print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
            images = batch['image']
            print(images.shape)

        # for i in range(3):
        #     image = images.squeeze(2)[0][i].permute(1, 2, 0).numpy()
        #     plt.imshow(image)
        #     plt.savefig(f"/home/yingzi/vlm/examples/few_shot_{i}.png")


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
