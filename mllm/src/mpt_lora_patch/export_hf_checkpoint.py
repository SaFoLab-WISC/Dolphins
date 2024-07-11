import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Based on https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py
# Note that this does NOT guard against no-op merges. I would suggest testing the output.

if len(sys.argv) != 4:
    print("Usage: python export_hf_checkpoint.py <source> <lora> <dest>")
    exit(1)

source_path = sys.argv[1]
lora_path = sys.argv[2]
dest_path = sys.argv[3]

base_model = AutoModelForCausalLM.from_pretrained(
    source_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    trust_remote_code=True,
)

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()
lora_model.train(False)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

base_model.save_pretrained(
    dest_path, state_dict=deloreanized_sd, max_shard_size="400MB"
)
