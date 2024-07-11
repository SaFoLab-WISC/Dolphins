import json

import sys

sys.path.append("..")


from dataset_utils.dolly_dataset import DollyDataset, TextPrompter
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn

class QuoraDataset(DollyDataset):
    """
    ```json
    [
        {"topic":
            "What is the step by step guide to invest in share market in india?",
         "input":
             "The conversation between human and AI assistant.\n[|Human|] I want to know the step by step guide to invest in share market in India.\n[|AI|] Sure, I can help with that. Firstly, you need to open a demat and trading account with a registered stockbroker.\n[|Human|] How do I find a registered stockbroker in India?\n[|AI|] You can visit the websites of National Stock Exchange (NSE) or Bombay Stock Exchange (BSE) to get a list of registered stockbrokers in India.\n[|Human|] What documents are required to open a demat and trading account?\n[|AI|] You will need to provide identity proof (PAN card), address proof, bank details and a passport size photograph to open a demat and trading account.\n[|Human|] How do I start trading once I have a demat and trading account?\n[|AI|] You can start trading by placing buy and sell orders for stocks through your stockbroker either online or offline.\n[|Human|] How do I track my investments?\n[|AI|] You can track your investments through your demat account. It will provide you with a consolidated view of all your investments in various stocks and other financial instruments.\n[|Human|] "},
    ]
    """
    def load_annotation(self, ann_path):
        self.annotation = json.load(open(ann_path, "r"))

    def process_text(self, ann):
        conversation = self.prompter.get_template_system() \
                       + self.prompter.get_template_topic().format(topic=ann["topic"])
        convs = ann["input"].split("\n")[1:-1]

        for i in range(0, len(convs) - 1, 2):
            question = convs[i].replace("[|Human|]", "").strip(" ")
            answer = convs[i+1].replace("[|AI|]", "").strip(" ")
            single_conv = self.prompter(question=question, response=answer, context=None, add_system=False)
            conversation += single_conv + "\n"

        return dict(conversation=conversation.strip("\n"))


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

    dataset = QuoraDataset(
        tokenizer,
        vis_processor=image_processor,
        ann_path=
        "/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/Text_Instructions/quora_chat_data_fliter.json"
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
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/quora.png")

        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        print(tokenizer.decode(labels[1][:sum(batch['attention_mask'][1])]))
        break