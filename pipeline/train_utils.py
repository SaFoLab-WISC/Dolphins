import time
from contextlib import suppress

import os
import torch
from tqdm import tqdm

def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == "fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return suppress


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


from huggingface_hub import hf_hub_download

def load_checkpoint(model, model_path, from_hub=False):
    if "openflamingo" in model_path:
        if from_hub:
            model_path = hf_hub_download(model_path, "checkpoint.pt")
        openflamingo = torch.load(model_path)
        for name, param in openflamingo.items():
            rename = name
            if "lang_encoder." in name:
                idx = name.index("lang_encoder.") + len("lang_encoder.")
                rename = rename[:idx] + "base_model.model." + rename[idx:]
            if rename in model.state_dict().keys():
                if "wte" in rename:
                    model.state_dict()[rename][:50280].copy_(param)
                else:
                    model.state_dict()[rename].copy_(param)
    else:
        if from_hub:
            model_path = hf_hub_download(model_path, "checkpoint_0.pt")
        model.load_state_dict(torch.load(model_path), strict=False)

    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (name, p,) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.base_model.model.old_decoder_blocks" in n)
        or ("lang_encoder.base_model.model.gated_cross_attn_layers" in n)
        or ("lang_encoder.base_model.old_decoder_blocks" in n)
        or ("lang_encoder.base_model.gated_cross_attn_layers" in n)
        or ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
        or ("word_embeddings" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, step, output_dir):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """

    model_state = model.state_dict()
    model_state = filter_state_dict_to_trainable(model, model_state)

    for k in model_state:
        model_state[k] = model_state[k].to(torch.float16).cpu()

    print(f"Saving checkpoint to {output_dir}/checkpoint.pt")
    torch.save(model_state, f"{output_dir}/checkpoint.pt")


