import sys

sys.path.append("..")

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable
import h5py
import cv2
import base64
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import LlamaTokenizer, CLIPImageProcessor

from dataset_utils.vqa_dataset import VQADataset, VisualPrompter
from dataset_utils.processors import DefaultTransform, FLAMINGO_MEAN, FLAMINGO_STD
from dataset_utils.collater import collate_fn
from conversation import get_conv_template


random.seed(42)

Q1_instruction = [
    "What is the current action of this vehicle?",
    "What is the vehicle doing right now in this video?",
    "What action is the vehicle performing in this video at the moment?",
    "Can you describe the vehicle's current activity in this video?",
    "What's happening with the vehicle in this video right now?",
    "At this moment in the video, what is the vehicle engaged in?",
    "What can you observe the vehicle doing in this video currently?",
    "How is the vehicle behaving at this point in the video?",
    "What is the ongoing action of the vehicle in the video?",
    "In this video, what action is the vehicle involved in at present?",
    "Describe the current state of the vehicle in this video.",
]

Q2_instruction = [
    "Why does this vehicle behave in this way?",
    "What is the reason behind this vehicle's behavior?",
    "Can you explain the cause of this vehicle's actions?",
    "What factors contribute to the way this vehicle is behaving?",
    "What's the rationale behind this vehicle's behavior?",
    "Why is the vehicle acting in this particular manner?",
    "What prompted the vehicle to behave like this?",
    "What circumstances led to this vehicle's behavior?",
    "What is the underlying cause of this vehicle's actions?",
    "For what reason is the vehicle exhibiting this behavior?",
    "What's driving the vehicle to behave in this way?",
]

Q3_instruction = [
    "Predict the speed and turning angle of the vehicle in the next second.",
    "Foresee the speed and turning angle of the vehicle in the following second.",
    "Anticipate the speed and turning angle of the vehicle in the subsequent second.",
    "Estimate the speed and turning angle of the vehicle in the next second.",
    "Project the speed and turning angle of the vehicle in the upcoming second.",
    "Forecast the speed and turning angle of the vehicle in the ensuing second.",
    "Envision the speed and turning angle of the vehicle in the next second.",
    "Expect the speed and turning angle of the vehicle in the following second.",
    "Presume the speed and turning angle of the vehicle in the subsequent second.",
    "Prognosticate the speed and turning angle of the vehicle in the next second.",
    "Calculate the speed and turning angle of the vehicle in the upcoming second.",
]


Q3_instruction_with_signals = [
    "Give you the current vehicle condition during this time. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>\n\nPredict the speed and turning angle of the vehicle in the next second.",
    "Based on the present condition of the ego vehicle during this time, foresee the speed and turning angle of the vehicle in the following second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Given the current vehicle condition during this time, anticipate the speed and turning angle of the vehicle in the subsequent second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Give you the current vehicle condition during this time. Estimate the speed and turning angle of the vehicle in the next second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Give you the current ego vehicle condition. Project the speed and turning angle of the vehicle in the upcoming second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "You are given the current vehicle condition during this time. Forecast the speed and turning angle of the vehicle in the ensuing second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "You are given the current vehicle condition during this time. Envision the speed and turning angle of the vehicle in the next second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "You are given you the ego vehicle's current condition. Expect the speed and turning angle of the vehicle in the following second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Given the current state of the vehicle, presume the speed and turning angle of the vehicle in the subsequent second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Based on the current state of the vehicle during this time, prognosticate the speed and turning angle of the vehicle in the next second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Based on the ego vehicle's current condition during this time, calculate the speed and turning angle of the vehicle in the upcoming second.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
]

Q4_instruction = [
    "If you are driving the vehicle in the video, please answer the following questions based on what you see and the current vehicle condition during this time.\nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "If you happen to be the one operating the vehicle in the video, kindly respond to the following queries based on your observations and the present condition of the ego vehicle at this moment. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Should you find yourself behind the wheel of the vehicle in the video, please address the following questions by taking into account your visual observations and the current state of the ego vehicle. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "In the event that you are responsible for driving the vehicle shown in the video, I request that you provide answers to the following inquiries based on your visual assessments and the current status of the ego vehicle. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "If you are in control of the vehicle featured in the video, I would appreciate it if you could respond to the following questions, taking into consideration what you can see and the ego vehicle's current condition. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
    "Assuming you are the driver of the vehicle depicted in the video, please offer your insights by answering the following questions, drawing from your visual observations and the current state of the vehicle. \nCurrent Vehicle Condition: Speed (m/s): <speed>; Accelerator: <accelerator>; Turn angle (degree): <turn_angle>",
]




class BDDXDataset(VQADataset):
    def __init__(self,
                 tokenizer,
                 vis_root,
                 ann_paths,
                 add_eos=True,
                 ignore_instruction=True,
                 max_seq_length=512,
                 max_num_images=5,
                 num_shots=5,
                 num_frames=16,
                 image_size=224,
                 task_type="mixed",
                 few_shot_ratio=0.5,
                 prompt_template="octopus",
                 mode="training", # validation, testing
                 signal_types=["speed", "accelerator", "turn_angle"],
                 ):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images
        self.vis_root = vis_root
        self.prompt_template = prompt_template
        self.num_shots = num_shots
        self.num_frames = num_frames
        self.signal_types = signal_types
        self.task_type = task_type
        self.few_shot_ratio = few_shot_ratio
        self.ann_paths = ann_paths
        self.mode = mode

        self.vis_processor = DefaultTransform(image_size=image_size)
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

        with open(ann_paths[0], "r") as f:
            self.annotation = json.load(f)

        self.load_video()
        self.load_car_infos()
        self.construct_examples()

        print(f"Training Dataset Length: {len(self.samples)}")
        # print(self.samples[0])
        # # print(self.samples[-5:])
        # #
        # for id in self.samples[0]['in_context_ids']:
        #     print(self.samples[id])

    def load_car_infos(self):
        info_path = os.path.join("/".join(self.ann_paths[0].split("/")[:-1]), "log")
        self.car_infos = {}
        for file in tqdm(os.listdir(info_path), desc="[Get Car Info]"):
            file_path = os.path.join(info_path, file)
            f = h5py.File(file_path, 'r')
            infos = {}
            vidName = file.split("_")[-1].split(".")[0].strip(" ")
            for key in f.keys():
                tmp = list(f[key])
                if "curvature" == key:
                    infos[key] = [round(item, 3) for item in tmp]
                    tmp = [item * (180 / math.pi) for item in tmp]
                    tmp = [round(item, 3) for item in tmp]
                    infos["turn_angle"] = tmp
                else:
                    tmp = [round(item, 3) for item in tmp]
                    infos[key] = tmp
            self.car_infos[vidName] = infos

    def load_video(self):
        self.video_data = {}
        with open(self.vis_root, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="[Load Video]"):
                line = line.strip('\n').split('\t')
                unqiue_id = self.mode + "_" + line[0].split("/")[-1] + "_" + line[0].split("/")[0]
                self.video_data[unqiue_id] = line[2:]

    def get_car_info(self, vidName, sTime, eTime):
        sensor_type_num = len(self.signal_types)
        sTime = int(sTime) if sTime == sTime else 0
        eTime = int(eTime) if eTime == eTime else 0
        # all_choices = ['course', 'curvature', 'accelerator', 'speed']
        time_length = int(eTime) - int(sTime)
        info_meta_data = {}
        for signal in self.signal_types:
            key_info = self.car_infos[vidName].get(signal, None)
            assert key_info is not None, "Control signals do not exist!"
            if eTime >= len(key_info): eTime = len(key_info) - 1
            while sTime >= eTime:
                if sTime > 0: sTime = sTime - 1
                elif eTime < len(key_info) - 1: eTime += 1
            assert sTime < eTime, f"{sTime}, {eTime}"
            info_meta_data[signal] = key_info[sTime: eTime]
            info_meta_data[f"{signal}_label"] = key_info[eTime]

        return info_meta_data

    def process_time(self, sTime, eTime):
        if str(sTime) == 'nan':
            eTime = int(eTime)
            sTime = eTime - 1
        if str(eTime) == "nan":
            sTime = int(sTime)
            eTime = sTime + 1
        return int(sTime), int(eTime)

    def convert_list2str(self, control_signals):
        control_signals = [str(item) for item in control_signals]
        control_signals = ", ".join(control_signals)
        return control_signals

    def prepare_conversation(self, line):
        if 'prompt' not in line['conversation'][0].keys(): return None
        if 'target' not in line['conversation'][0].keys(): return None
        line['prompt'] = line['conversation'][0]['prompt']
        line['target'] = line['conversation'][0]['target']
        line['task_type'] = "conversation"
        return line

    def prepare_action_QA(self, line, is_single=False):
        ####  action  ####
        instruction = [Q1_instruction[0]] if self.mode != "training" else Q1_instruction
        action_example = line.copy()
        action_example['prompt'] = random.sample(instruction, 1)[0]
        target = str(line['action']).capitalize()
        if target[-1] >= 'a' and target[-1] <= 'z': target += "."
        action_example['target'] = target
        action_example['task_type'] = "action"
        return action_example

    def prepare_justification_QA(self, line, is_single=False):
        ####  justification  ####
        instruction = [Q2_instruction[0]] if self.mode != "training" else Q2_instruction
        justification_example = line.copy()
        justification_example['prompt'] = random.sample(instruction, 1)[0]
        target = str(line['justification']).capitalize()
        if target[-1] >= 'a' and target[-1] <= 'z': target += "."
        justification_example['target'] = target
        justification_example['task_type'] = "justification"
        return justification_example

    def prepare_signals_QA(self, line, is_single=True):
        ####  predict control signals ####

        control_signal_example = line.copy()
        if is_single:
            instruction = [Q3_instruction_with_signals[0]] if self.mode != "training" else Q3_instruction_with_signals
            control_signal_example['prompt'] = random.sample(instruction, 1)[0]
        else:
            instruction = [Q3_instruction[0]] if self.mode != "training" else Q3_instruction
            control_signal_example['prompt'] = random.sample(instruction, 1)[0]
        control_signal_example['target'] = f"Speed: {line['control_signals']['speed_label']}; Turn angle: {line['control_signals']['turn_angle_label']}"
        control_signal_example['prompt'] = control_signal_example['prompt'].replace(
            "<speed>", self.convert_list2str(line['control_signals']['speed']),
        ).replace(
            "<accelerator>", self.convert_list2str(line['control_signals']['accelerator'])
        ).replace(
            "<turn_angle>", self.convert_list2str(line['control_signals']['turn_angle'])
        )
        control_signal_example['task_type'] = "control_signals"
        return control_signal_example

    def prepare_example(self, line, task_type):
        if task_type == 'conversation':
            return self.prepare_conversation(line)
        if task_type == 'action':
            return self.prepare_action_QA(line)
        if task_type == 'justification':
            return self.prepare_justification_QA(line)
        if task_type == 'control_signals':
            return self.prepare_signals_QA(line)

    def construct_examples(self):
        """
        example format:

        prompt:
        target:
        vidName:
        sTime:
        eTime:
        speed:
        turn_angle:
        course:

        """

        self.single_fixed_QA, self.multi_fixed_QA, self.conversation = [], [], []
        for uid, line in tqdm(self.annotation.items(), desc=f"[ChatGPT Conversation]"):
            if line['conversation'][0] == {}  or \
                    (str(line['sTime']) == "nan" and str(line['eTime']) == "nan"): continue
            line['unique_id'] = uid
            line['sTime'], line['eTime'] = self.process_time(line['sTime'], line['eTime'])
            self.conversation.append(self.prepare_conversation(line))

        num_fixed_QA = 0
        num_examples_split = 1 if self.mode != "training" else 2
        for uid, line in tqdm(self.annotation.items(), desc="[Fixed QA]"):
            if str(line['action']) == "nan" or \
                    str(line['justification']) == "nan" or \
                    (str(line['sTime']) == "nan" and str(line['eTime']) == "nan"): continue
            line['unique_id'] = uid
            line['sTime'], line['eTime'] = self.process_time(line['sTime'], line['eTime'])
            line['control_signals'] = self.get_car_info(line['vidName'], line['sTime'], line['eTime'])
            ##### single fixed QA #####
            num_fixed_QA += 1
            if num_fixed_QA <= len(self.annotation.keys()) // num_examples_split:
                self.single_fixed_QA.append(self.prepare_action_QA(line))
                self.single_fixed_QA.append(self.prepare_justification_QA(line))
                self.single_fixed_QA.append(self.prepare_signals_QA(line))

            ##### multiple fixed QA #####
            else:
                action_example, \
                justification_example, \
                control_signal_example \
                    = self.prepare_action_QA(line), \
                      self.prepare_justification_QA(line), \
                      self.prepare_signals_QA(line, is_single=False)
                instruction = random.sample(Q3_instruction, 1)[0].replace(
                    "<speed>", self.convert_list2str(control_signal_example['control_signals']['speed'])
                ).replace(
                    "<accelerator>", self.convert_list2str(control_signal_example['control_signals']['accelerator'])
                ).replace(
                    "<turn_angle>", self.convert_list2str(control_signal_example['control_signals']['turn_angle'])
                )
                action_example['prompt'] = instruction + "\n" + action_example['prompt']

                dialog_hist = []
                dialog_hist.append({"q": action_example['prompt'], "a": action_example['target']})
                dialog_hist.append({"q": justification_example['prompt'], "a": justification_example['target']})
                control_signal_example['dialog_hist'] = dialog_hist
                del control_signal_example['in_context_ids']
                self.multi_fixed_QA.append(control_signal_example)

        self.samples = []
        if self.task_type == "mixed":
            self.samples.extend(self.conversation)
            self.samples.extend(self.single_fixed_QA)
            self.samples.extend(self.multi_fixed_QA)
        elif self.task_type == "fixedQA":
            self.samples.extend(self.single_fixed_QA)
            self.samples.extend(self.multi_fixed_QA)
        elif self.task_type == "convs":
            self.samples.extend(self.conversation)
        else:
            self.samples.extend(self.single_fixed_QA)

        for item in self.samples:
            if 'in_context_ids' in item.keys():
                item['in_context_ids'] = [idx for idx in item['in_context_ids'] if idx in self.video_data.keys()]

        self.samples = [item for item in self.samples if item['unique_id'] in self.video_data.keys()]


    def img_from_base64(self, imagestring):
        try:
            jpgbytestring = base64.b64decode(imagestring)
            nparr = np.frombuffer(jpgbytestring, np.uint8)
            r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return r
        except ValueError:
            return None

    def extract_frames(self, unique_id, num_frames=16):
        video_np_array = []
        for i in range(len(self.video_data[unique_id])):
            video_np_array.append(self.img_from_base64(self.video_data[unique_id][i]))
        frames = np.array(video_np_array)

        def sampling(start, end, n):
            if n == 1:
                return [int(round((start + end) / 2.))]
            if n < 1:
                raise Exception("behaviour not defined for n<2")
            step = (end - start) / float(n - 1)
            return [int(round(start + x * step)) for x in range(n)]

        frames_extracted = []

        for i in sampling(0, len(frames) - 1, self.num_frames):
            frame = frames[i]
            frame = Image.fromarray(frame).convert("RGB")
            frame = self.vis_processor(frame)
            frames_extracted.append(frame)

        return torch.stack(frames_extracted, dim=0)

    def process_video(self, ann):
        """
        videos : B N F C W H
        """

        frame_list = []
        if 'in_context_ids' in ann.keys():
            for i, uid in enumerate(ann['in_context_ids']):
                frames = self.extract_frames(uid, self.num_frames)
                frame_list.append(frames)

        frames = self.extract_frames(ann['unique_id'], self.num_frames)
        frame_list.append(frames)
        return torch.stack(frame_list, dim=0)

    def process_image(self, images):
        keep_ixs = range(min(len(images), self.max_num_images))
        images_tensors = images[keep_ixs]
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), self.num_frames, 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float)
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        image = images_tensors
        return image

    def process_text(self, ann):
        text = ""
        if "dialog_hist" not in ann.keys():
            if random.random() > self.few_shot_ratio:
                for uid in ann['in_context_ids']:
                    example = self.prepare_example(self.annotation[uid], ann['task_type'] )
                    if example is not None:
                        text += "USER: <image> is a driving video. " + example['prompt']
                        text += " GPT: " + example['target'] + "<|endofchunk|>"
            text += "USER: <image> is a driving video. " + ann['prompt']
            if self.mode == "training":
                text += " GPT: <answer>" + ann['target'] + "<|endofchunk|>"
            else:
                text += " GPT: <answer>"
        else:
            prefix = "<image> is a driving video. "
            for i, convs in enumerate(ann['dialog_hist']):
                text += "USER: " + prefix + convs['q']
                text += " GPT: " + convs['a'] + "<|endofchunk|>"
                prefix = ""
            text += "USER: " + ann['prompt']
            if self.mode == "training":
                text += " GPT: <answer>" + ann['target'] + "<|endofchunk|>"
            else:
                text += " GPT: <answer>"
        return dict(conversation=text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ann = self.samples[index]
        if 'in_context_ids' in ann.keys():
            if len(ann['in_context_ids']) > self.num_shots:
                ann['in_context_ids'] = ann['in_context_ids'][:self.num_shots]
        frames = self.process_video(ann)
        image = self.process_image(frames)
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(unique_id = ann['unique_id'])
        res.update(image = image)
        res.update(text = text)
        if self.mode != "training":
            res.update(action = ann['action'])
            res.update(justification = ann['justification'])
        return res

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True,
                             max_length=self.max_seq_length, add_special_tokens=False)
        if self.mode == "training":
            res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
            res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
        else:
            res["input_ids"] = res["input_ids"]
            res["attention_mask"] = res["attention_mask"]
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

    dataset = BDDXDataset(
        tokenizer=tokenizer,
        vis_root="/home/scratch.sysarch_nvresearch/chaowei/BDD-X/data/video/training_32frames_img_size256.img.tsv",
        ann_paths=["/home/scratch.sysarch_nvresearch/chaowei/BDD-X/data/drive_caption/captions_BDDX_clean.json",],
        max_seq_length=1024,
        mode="training",
        num_shots=3,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=16,
        pin_memory=True,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        # shuffle=True,
        collate_fn=dataset.collater,
    )

    for batch in tqdm(train_dataloader):
        print(batch['unique_id'][0])
        batch = batch['net_input']
        for k, v in batch.items():
            if isinstance(v, list):
                print(k, len(v))
                continue
            print(k, v.shape)

        print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))

        # images = batch['image'][0]
        # print(images.shape)
        # # 4 6 16 3 224 224
        # for i in range(images.shape[0]):
        #     for j in range(images.shape[1]):
        #         image = images[i][j].permute(1, 2, 0).numpy() * 255
        #         image = Image.fromarray(np.uint8(image))
        #         image.save(
        #             f"/home/scratch.chaoweix_nvresearch/visual_instruction/visual-instruction-meta-learning/workspace/data/examples/examples/BDD-X_{i}_{j}.png")

        break
