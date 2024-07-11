import sys

sys.path.append("..")

import copy
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

from dataset_utils.vqa_dataset import VQADataset
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn


TEMPLATE = {
    "system": "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n",
    "topic": "Human: I will give you a specific topic, and let's have some discussions around this topic. The topic is: {topic} AI: <answer> Of course, I am very happy to do so. <|endofchunk|>",
    "prompt_convs": "Human: {question} AI: <answer> {response} <|endofchunk|>",
    "prompt_convs_with_context": "Human: {question} Human: {context} AI: <answer> {response} <|endofchunk|>",
}


class TextPrompter:

    def __call__(self, question, response, context=None, add_system=True):
        if add_system:
            res = TEMPLATE["system"]
        else:
            res = ""
        if context:
            res += TEMPLATE["prompt_convs_with_context"].format(question=question, context=context, response=response)
        else:
            res += TEMPLATE["prompt_convs"].format(question=question, response=response)
        return res

    def get_template_system(self):
        return TEMPLATE["system"]

    def get_template_topic(self):
        return TEMPLATE["topic"]


class DollyDataset(Dataset):
    """Each line of the annotation file is a json object with the following fields:

    {
        "instruction": "What is a dispersive prism?",
        "context": "In optics, a dispersive prism is an optical prism that is used to disperse light, that is, to separate light into its spectral components (the colors of the rainbow). Different wavelengths (colors) of light will be deflected by the prism at different angles.[1] This is a result of the prism material's index of refraction varying with wavelength (dispersion). Generally, longer wavelengths (red) undergo a smaller deviation than shorter wavelengths (blue). The dispersion of white light into colors by a prism led Sir Isaac Newton to conclude that white light consisted of a mixture of different colors.",
        "response": "A dispersive prism is an optical prism that disperses the light's different wavelengths at different angles. When white light is shined through a dispersive prism it will separate into the different colors of the rainbow.",
        "category": "summarization"
    }

    """

    def __init__(self,
                 tokenizer,
                 ann_path: str,
                 add_eos=True,
                 ignore_instruction=True,
                 max_seq_length=512,
                 max_num_images=5,
                 vis_processor=None,
                 **kwargs):
        """
        ann_path (string): directory to store the annotation file
        """
        assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: LlamaTokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_num_images = max_num_images

        self.annotation = []

        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.load_annotation(ann_path)

        self.bos_item = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([self.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

        self.vis_processor = vis_processor
        self.prompter = TextPrompter()


    def load_annotation(self, ann_path):
        self.annotation = []
        for line in open(ann_path, "r").readlines():
            self.annotation.append(json.loads(line))

    def __len__(self):
        return len(self.annotation)

    def process_image(self, ann):
        image = Image.new('RGB', (224, 224), (0, 0, 0))
        image = [self.vis_processor(image)]

        region_images = torch.stack(image, dim=0)
        keep_ixs = range(min(len(region_images), self.max_num_images))
        images_tensors = region_images[keep_ixs]

        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (self.max_num_images - len(images_tensors), 3, self.vis_processor.image_size,
                 self.vis_processor.image_size), dtype=torch.float )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

        image = images_tensors.unsqueeze(1)
        return image

    def process_text(self, ann):
        instruction = ann["instruction"]
        context =  ann["context"] if len(ann["context"]) > 0 else None
        response = ann["response"]
        conversation = self.prompter(question=instruction, response=response, context=context)
        return dict(conversation=conversation)

    def tokenize(self, text):
        res = self.tokenizer(text["conversation"], return_tensors="pt", padding="do_not_pad", truncation=True, max_length=self.max_seq_length, add_special_tokens = False)
        res["input_ids"] = torch.cat([self.bos_item, res["input_ids"].squeeze(0), self.eos_item]).unsqueeze(0)
        res["attention_mask"] = torch.cat([self.bos_mask, res["attention_mask"].squeeze(0), self.eos_mask]).unsqueeze(0)
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

        return collate_fn(samples,
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id,
                          left_pad=self.tokenizer.padding_side == "left"
                          )


def build_dolly_dataset(
    tokenizer,
    ann_path="data/dolly/databricks-dolly-15k.jsonl",
    **kwargs,
):
    return DollyDataset(
        tokenizer=tokenizer,
        ann_path=ann_path,
        **kwargs,
    )


from transformers import LlamaTokenizer
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

    dataset = build_dolly_dataset(
        tokenizer,
        vis_processor=image_processor,
        ann_path=
        "/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/Text_Instructions/databricks-dolly-15k.jsonl"
    )



    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        #sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    from tqdm import tqdm

    for batch in tqdm(train_dataloader):
        batch = batch['net_input']
        break

    start_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

    for batch in train_dataloader:
        batch = batch['net_input']
        input_ids = batch["input_ids"]

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = 0
        labels[:, 0] = 0

        for i in range(labels.shape[0]):
            # remove loss for any token before human token
            label_idx = 0
            while (
                    label_idx < labels.shape[1] and labels[i][label_idx] != start_token_id
            ):
                labels[i][label_idx] = 0
                label_idx += 1

            labels[i][label_idx] = 0

        labels[labels == answer_token_id] = 0
        labels[labels == media_token_id] = 0

        images = batch['image']
        image = images.squeeze(2)[0][0].permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/dolly.png")

        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        print(tokenizer.decode(labels[1][:sum(batch['attention_mask'][1])]))
        break