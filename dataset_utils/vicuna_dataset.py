import json

import sys

sys.path.append("..")


from dataset_utils.dolly_dataset import DollyDataset, TextPrompter
from dataset_utils.processors import DefaultTransform
from dataset_utils.collater import  collate_fn

class ShareGPTDataset(DollyDataset):
    """
    ```json
    [
        {
            "id": "A5AbcES_0",
            "conversations": [
              {
                "from": "human",
                "value": "In Java, I want to replace string like \"This is a new {object} at {place}\" with a Map, {object: \"student\", \"point 3, 4\"}, and get a result \"This is a new student at point 3, 4\". How can I do?"
              },
              {
                "from": "gpt",
                "value": "You can use the `String.format()` method in Java to replace placeholders in a string with values from a map. Here's an example code snippet that demonstrates how you can achieve this:\n```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic class StringReplaceExample {\n    public static void main(String[] args) {\n        String input = \"This is a new {object} at {place}\";\n        Map<String, String> replacements = new HashMap<>();\n        replacements.put(\"object\", \"student\");\n        replacements.put(\"place\", \"point 3, 4\");\n\n        String output = replacePlaceholders(input, replacements);\n        System.out.println(output);\n    }\n\n    public static String replacePlaceholders(String input, Map<String, String> replacements) {\n        for (Map.Entry<String, String> entry : replacements.entrySet()) {\n            String placeholder = \"{\" + entry.getKey() + \"}\";\n            String replacement = entry.getValue();\n            input = input.replace(placeholder, replacement);\n        }\n        return input;\n    }\n}\n```\nIn this example, we define the input string as \"This is a new {object} at {place}\", and create a `HashMap` called `replacements` that maps the placeholders \"object\" and \"place\" to their corresponding values \"student\" and \"point 3, 4\". We then pass these values to the `replacePlaceholders()` method, which iterates over the entries in the `replacements` map and replaces each placeholder in the input string with its corresponding value using the `replace()` method. Finally, the `replacePlaceholders()` method returns the modified string, which is printed to the console.\n\nThe output of this program will be:\n```csharp\nThis is a new student at point 3, 4\n```\nNote that you can modify the `replacements` map to include additional placeholders and their corresponding values, and the `replacePlaceholders()` method will automatically replace them in the input string."
              }
            ]
        },
    ]
    """
    def load_annotation(self, ann_path):
        self.annotation = json.load(open(ann_path, "r"))

    def process_text(self, ann):
        convs = ann["conversations"]
        conversation = self.prompter.get_template_system()
        for i in range(0, len(convs) - 1, 2):
            question = convs[i]["value"]
            answer = convs[i+1]["value"]
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

    dataset = ShareGPTDataset(
        tokenizer,
        vis_processor=image_processor,
        ann_path=
        "/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/workspace/datasets/Text_Instructions/ShareGPT_V3_fliter.json"
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
        plt.savefig("/home/scratch.chaoweix_nvresearch/visual_instruction/vlm/examples/vicuna.png")

        print(tokenizer.decode(batch["input_ids"][1][:sum(batch['attention_mask'][1])]))
        print(tokenizer.decode(labels[1][:sum(batch['attention_mask'][1])]))
        break