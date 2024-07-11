import contextlib

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from collections import defaultdict

def collate_fn_for_LLaVA(samples):
    batch = {
        "net_input": defaultdict(list),
    }
    if samples[0].get("task_name", None) is not None:
        batch["task_name"] = [sample["task_name"] for sample in samples]

    if samples[0].get("image_id", None) is not None:
        batch["image_id"] = [sample["image_id"] for sample in samples]

    if samples[0].get("unique_id", None) is not None:
        batch["unique_id"] = [sample["unique_id"] for sample in samples]

    if samples[0].get("input_ids", None) is not None:
        for sample in samples:
            batch['net_input']["input_ids"].append(sample["input_ids"])
        batch['net_input']["input_ids"] = torch.stack(batch['net_input']["input_ids"], dim=0).squeeze(1)

    if samples[0].get("images", None) is not None:
        for sample in samples:
            batch['net_input']["images"].append(sample["images"])
        batch['net_input']["images"] = torch.stack(batch['net_input']["images"], dim=0).squeeze(1)

    return batch


def collate_fn_for_instructBLIP(samples):
    batch = {
        "net_input": defaultdict(list),
    }

    if samples[0].get("task_name", None) is not None:
        batch["task_name"] = [sample["task_name"] for sample in samples]

    if samples[0].get("image_id", None) is not None:
        batch["image_id"] = [sample["image_id"] for sample in samples]

    if samples[0].get("unique_id", None) is not None:
        batch["unique_id"] = [sample["unique_id"] for sample in samples]


    if samples[0].get("inputs", None) is not None:
        for sample in samples:
            for k, v in sample["inputs"].items():
                batch['net_input'][k].append(v)

        for k, v in batch['net_input'].items():
            batch['net_input'][k] = torch.stack(batch['net_input'][k], dim=0)

    return batch



def collate_fn(samples, pad_idx, eos_idx, left_pad):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, left_pad=None):
        res = collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )
        return res

    src_tokens = merge("input_ids", pad_idx=pad_idx, left_pad=left_pad)
    src_tokens_masks = merge("attention_mask", pad_idx=0, left_pad=left_pad)

    batch = {
        "net_input": {
            "input_ids": src_tokens,
            "attention_mask": src_tokens_masks,
        },
    }

    if samples[0].get("image", None) is not None:
        batch["net_input"]["image"] = torch.stack(
            [sample["image"] for sample in samples], dim=0)

    if samples[0].get("task_name", None) is not None:
        batch["task_name"] = [sample["task_name"] for sample in samples]

    if samples[0].get("image_id", None) is not None:
        batch["image_id"] = [sample["image_id"] for sample in samples]

    if samples[0].get("unique_id", None) is not None:
        batch["unique_id"] = [sample["unique_id"] for sample in samples]
        
    if samples[0].get("text", None) is not None:
        batch["text"] = [sample["text"] for sample in samples]
    
    if samples[0].get("action", None) is not None:
        batch["action"] = [sample["action"] for sample in samples]
        
    if samples[0].get("justification", None) is not None:
        batch["justification"] = [sample["justification"] for sample in samples]

    # image grounded
    if samples[0].get("patch_image", None) is not None:
        batch["net_input"]["patch_images"] = torch.stack(
            [sample["patch_image"] for sample in samples], dim=0)
    if samples[0].get("patch_mask", None) is not None:
        batch["net_input"]["patch_masks"] = torch.cat(
            [sample["patch_mask"] for sample in samples])
    # image generation
    if samples[0].get("code_mask", None) is not None:
        batch["net_input"]["code_masks"] = torch.cat(
            [sample["code_mask"] for sample in samples])
    if samples[0].get("code_image", None) is not None:
        batch["code_images"] = torch.cat(
            [sample["code_image"] for sample in samples])
    # For classification tasks (i.e., VQA, SNLI-VE, GLUE)
    if samples[0].get("conf", None) is not None:
        batch["conf"] = torch.cat([s["conf"] for s in samples], dim=0)
    if samples[0].get("ref_dict", None) is not None:
        batch["ref_dict"] = np.array([s["ref_dict"] for s in samples])
    if samples[0].get("constraint_mask", None) is not None:
        batch["constraint_masks"] = merge("constraint_mask")
    if samples[0].get("decoder_prompt", None) is not None:
        batch["decoder_prompts"] = np.array(
            [s["decoder_prompt"].tolist() for s in samples])
    # For detection and visual grounding
    if samples[0].get("w_resize_ratio", None) is not None:
        batch["w_resize_ratios"] = torch.stack(
            [s["w_resize_ratio"] for s in samples], dim=0)
    if samples[0].get("h_resize_ratio", None) is not None:
        batch["h_resize_ratios"] = torch.stack(
            [s["h_resize_ratio"] for s in samples], dim=0)
    if samples[0].get("region_coord", None) is not None:
        batch["region_coords"] = torch.stack(
            [s["region_coord"] for s in samples], dim=0)

    return batch


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.shape[1] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    res = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor(v.squeeze(0), res[i][size - v.shape[1]:] if left_pad else res[i][:v.shape[1]])

    return res
