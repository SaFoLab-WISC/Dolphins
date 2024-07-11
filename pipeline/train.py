""" Main training script """

import sys

sys.path.append("..")

import argparse
import glob
import os
import gc
import random
import logging
import numpy as np
import wandb
from tqdm import tqdm
import time
import math
from configs.dataset_config import DATASET_CONFIG
from configs.lora_config import openflamingo_tuning_config
from configs.prompt_tuning_config import pt_config
from PIL import Image

import datasets
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
import deepspeed
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, PeftConfig, PeftModel, PromptTuningConfig

import transformers
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, CLIPImageProcessor,
                          LlamaForCausalLM, LlamaTokenizer, get_scheduler,
                          SchedulerType)

from train_utils import (
    AverageMeter,
    get_autocast,
    get_cast_dtype,
    get_checkpoint,
    load_checkpoint,
    save_checkpoint
)

from dataset_utils.arguments import add_data_args
from dataset_utils.processors import DefaultTransform, RegionTransform
from dataset_utils.builder import build_dataset
from mllm.src.factory import create_model_and_transforms

from huggingface_hub import hf_hub_download
import torch

logger = get_logger(__name__)
# os.environ["WANDB_MODE"] = "offline"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the final model.")
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.")
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help=
        "The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--llm_path",
        type=str,
        default="anas-awadalla/mpt-7b",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Path to image data.",
        required=False,
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="Path to dataset config.",
        required=False,
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        help="Path to lora config.",
        required=False,
    )
    parser.add_argument(
        "--peft_model_id",
        type=str,
        default=None,
        help="Path to peft model id.",
        required=False,
    )
    parser.add_argument(
        "--task_type",
        choices=["caption", "qa"],
        default="qa",
        help="the type of task.",
    )
    parser.add_argument(
        "--instruction_type",
        default="llava",
        help="the type of instructions.",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["all", "sic", "ric", "mic", "llava", "llava_dial", "minigpt4", "meme_cap", "ocr_vqa", "gqa", "vsr", "icon_qa", "aokvqa", "dolly", "alpaca_gpt4", "instruct"],
        default="all",
        help="the type of dataset for training.",
    )
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained",
                        default="openai",
                        type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help=
        "how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default=None,
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter_9b",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation",
                        action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="log loss every n steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    # Sum of gradient optimization batch size
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help=
        "path to huggingface model or model identifier from local path or huggingface.co",
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help=
        "path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--overwrite_checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--multi_instruct_path",
        type=str,
        help=
        "path to multi_instruct dataset, this should be a glob pattern such as vision_language_examples.tsv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument("--loss_multiplier_multi_instruct",
                        type=float,
                        default=1.0)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train_num_samples", type=int)
    parser.add_argument("--dataset_resampled", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_prompt_tuning", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend",
                        default="nccl",
                        type=str,
                        help="distributed backend")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--only_attend_immediate_media",
        default=False,
        action="store_true",
        help="No effect for a single image,  but will work for multi images reasoning.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help=
        "Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--load_hf_model", default=False, action="store_true")
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=
        ('The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
         ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
         "Only applicable when `--with_tracking` is passed."),
    )

    parser = add_data_args(parser)
    args = parser.parse_args()

    print(args)

    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else [])
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    train_dataset_config = DATASET_CONFIG[args.instruction_type]
    if args.dataset_type != "all":
        dataset_type_to_id = {}
        for i in range(len(train_dataset_config)):
            dataset_type_to_id[train_dataset_config[i]['type']] = i
        train_dataset_config = train_dataset_config[dataset_type_to_id[args.dataset_type]]

    if args.use_lora:
        lora_config = openflamingo_tuning_config
        peft_config = LoraConfig(**lora_config)
        model_name = "octopus"
    elif args.use_prompt_tuning:
        pt_config['tokenizer_name_or_path'] = args.llm_path
        peft_config = PromptTuningConfig(**pt_config)
        model_name = "octopus"
    else:
        peft_config = None
        model_name = "openflamingo"

    if args.model_name_or_path is not None:
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            clip_vision_encoder_cache_dir="/home/yingzi/models/openai-clip-vit-large-patch14",
            lang_encoder_path=args.llm_path,
            tokenizer_path=args.llm_path,
            cross_attn_every_n_layers=args.cross_attn_every_n_layers,
            use_peft=True if peft_config is not None else False,
            peft_config=peft_config,
        )
        model = load_checkpoint(model, args.model_name_or_path, args.load_hf_model)
    else:
        raise NotImplementedError
        
    print(train_dataset_config)
    train_dataset = build_dataset(
        dataset_config=train_dataset_config,
        tokenizer=tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.workers,
        # sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        shuffle=True,
        collate_fn=train_dataset.collater,
    )

    print(len(train_dataset))

    if accelerator.is_main_process:
        for batch in train_dataloader:
            batch = batch['net_input']
            for k, v in batch.items():
                if isinstance(v, list):
                    print(k, len(v))
                    continue
                print(k, v.shape)

            print(tokenizer.decode(batch["input_ids"][0][:sum(batch['attention_mask'][0])]))
            break

    params_name = []
    params_pt_name = []
    params_no_decay_name = []
    def get_grouped_params(model):
        params_with_wd, params_with_pt, params_without_wd = [], [], []

        def apply_decay(x):
            return ("gated_cross_attn_layer" in x and "ff_gate" not in x
                    and "attn_gate" not in x and "norm" not in x
                    and "bias" not in x) or ("lora" in x)

        for n, p in model.named_parameters():
            if p.requires_grad:
                if "prompt_encoder" in n:
                    params_pt_name.append(n)
                    params_with_pt.append(p)
                elif apply_decay(n):
                    params_name.append(n)
                    params_with_wd.append(p)
                else:
                    params_no_decay_name.append(n)
                    params_without_wd.append(p)

        return [
            {
                "params": params_with_wd,
                "weight_decay": args.weight_decay
            },
            {
                "params": params_without_wd,
                "weight_decay": 0.0
            },
            {
                "params": params_with_pt,
                "lr": 0.001,
                "weight_decay": 0.001,
            },
        ]

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps < 0:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=round(args.warmup_ratio * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    all_param, trainable_params = 0, 0
    if accelerator.is_main_process:
        for n, p in model.named_parameters():
            all_param += p.numel()
            if p.requires_grad:
                print(n, p.size())
                trainable_params += p.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    experiment_config = vars(args)
    experiment_config["lr_scheduler_type"] = experiment_config[
        "lr_scheduler_type"].value
    accelerator.init_trackers("vlm", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total warmup steps = {int(args.warmup_ratio * args.max_train_steps)}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(args.max_train_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    start_token_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]

    for epoch in range(starting_epoch, args.num_train_epochs):

        model.train()
        total_loss = 0
        losses = []
        cast_dtype  = get_cast_dtype(accelerator.mixed_precision)

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            images = None
            # 4, 5, 1, 3 , 224,224
            if "image" in batch["net_input"].keys():
                images = (batch["net_input"]["image"].to(dtype=cast_dtype))
            input_ids = batch["net_input"]["input_ids"]
            attention_mask = batch["net_input"]["attention_mask"]

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[labels == tokenizer.eos_token] = -100
            labels[:, 0] = -100

            if "openflamingo" not in args.instruction_type:
                for i in range(labels.shape[0]):
                    endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
                    media_idxs = torch.where(labels[i] == media_token_id)[0]
                    # remove loss for any token the before the first <answer>
                    token_idx = 0
                    while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                        labels[i][token_idx] = -100
                        token_idx += 1

            labels = labels.to(input_ids.device)
            labels[labels == answer_token_id] = -100
            labels[labels == media_token_id] = -100

            media_locations = None
            if args.use_prompt_tuning:
                prefix_labels = torch.full((input_ids.shape[0], peft_config.num_virtual_tokens), -100).to(input_ids.device)
                media_locations = torch.cat((prefix_labels, input_ids), dim=1) == media_token_id

            with accelerator.accumulate(model):
                loss = model(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    media_locations=media_locations,
                )[0]

                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - loss: {loss:.4f}")

                total_loss += loss.detach().float()
                losses.append(loss.detach().float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                accumulate_loss = torch.tensor(losses)
                # filter out nan
                accumulate_loss = accumulate_loss[~torch.isnan(accumulate_loss)]
                train_loss = torch.mean(accumulate_loss)

                losses = []
                accelerator.log(
                    {
                        "train_loss": train_loss.item(),
                        "step": completed_steps,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                    },
                    step=completed_steps,
                )
      
                if completed_steps in [540, 720]:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, args.instruction_type + "_" + args.dataset_type, output_dir)
                    if accelerator.is_main_process:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_checkpoint(unwrapped_model, completed_steps, output_dir)

                        gc.collect()
                        torch.cuda.empty_cache()

                if completed_steps >= args.max_train_steps:
                    break

    accelerator.end_training()

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, args.instruction_type + "_" + args.dataset_type)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            try:
                os.makedirs(output_dir)
            except OSError:
                pass
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped_model, 0, output_dir)
            tokenizer.save_pretrained(args.output_dir, args.dataset_type)

if __name__ == "__main__":
    main()
