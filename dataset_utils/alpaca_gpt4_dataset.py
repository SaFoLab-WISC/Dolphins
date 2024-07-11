import json

import sys

sys.path.append("..")


from dataset_utils.dolly_dataset import DollyDataset, TextPrompter
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn

class AlpacaGPT4Dataset(DollyDataset):
    """
    ```json
    [
        {
            "instruction": "Identify the odd one out.",
            "input": "Twitter, Instagram, Telegram",
            "output": "The odd one out is Telegram. Twitter and Instagram are social media platforms mainly for sharing information, images and videos while Telegram is a cloud-based instant messaging and voice-over-IP service."
        },
    ]
    """
    def load_annotation(self, ann_path):
        self.annotation = json.load(open(ann_path, "r"))

    def process_text(self, ann):
        instruction = ann["instruction"]
        input = str(ann["input"]) if len(str(ann["input"])) >= 0 else None
        output = ann["output"]
        conversation = self.prompter(question=instruction, response=output, context=input)
        return dict(conversation=conversation)


from transformers import LlamaTokenizer
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

    dataset = AlpacaGPT4Dataset(
        tokenizer,
        vis_processor=image_processor,
        ann_path=
        "/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/Text_Instructions/alpaca_data.json"
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
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/alpaca.png")

        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        print(tokenizer.decode(labels[1][:sum(batch['attention_mask'][1])]))
        break