
viml_instruct_dataset = {
    "vis_root": "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
    "ann_paths": "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/train.jsonl",
    "cot_path": "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/train_cot.json",
}

inst_datasets = [
    dict(
        type="instruct",
        vis_root=viml_instruct_dataset['vis_root'],
        ann_paths=[
            viml_instruct_dataset['ann_paths'],
        ],
        max_seq_length=1024,
        template="octopus",
        mode="train",
        # with_exemplars=True,
        # cot_path="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/train_cot.json",
    ),
]

inst_with_icl_datasets = [
    dict(
        type="instruct",
        vis_root=viml_instruct_dataset['vis_root'],
        ann_paths=[
            viml_instruct_dataset['ann_paths'],
        ],
        max_seq_length=1024,
        template="octopus",
        mode="train",
        with_exemplars=True,
        #cot_path="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/train_cot.json",
    ),
]

inst_with_cot_datasets = [
    dict(
        type="instruct",
        vis_root=viml_instruct_dataset['vis_root'],
        ann_paths=[
            viml_instruct_dataset['ann_paths'],
        ],
        max_seq_length=1024,
        template="octopus",
        mode="train",
        # with_exemplars=True,
        cot_path=viml_instruct_dataset['cot_path'],
    ),
]

inst_with_icl_cot_datasets = [
    dict(
        type="instruct",
        vis_root=viml_instruct_dataset['vis_root'],
        ann_paths=[
            viml_instruct_dataset['ann_paths'],
        ],
        max_seq_length=1024,
        template="octopus",
        mode="train",
        with_exemplars=True,
        cot_path=viml_instruct_dataset['cot_path'],
    ),
]

alignment_datasets = [
    dict(
        type="llava",
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/llava/detail_23k.json",
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/llava/complex_reasoning_77k.json",
        ],
        max_seq_length=256,
        sample=50000,
    ),
    dict(
        type="llava_dial",
        vis_root="/home/yingzi/MultiInstruct/MSCOCO2014/train2014",
        ann_paths=["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/llava/conversation_58k.json"],
        max_seq_length=256,
        sample=50000,
    ),
    dict(
        type="svit_bbox",
        vis_root="/home/yingzi/MultiInstruct/visual_genome",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/svit/complex_reasoning_bbox.jsonl",
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/svit/detail_description_bbox.jsonl",
        ],
        max_seq_length=256,
        sample=50000,
    ),
    dict(
        type="lrv",
        vis_root="/home/yingzi/MultiInstruct/visual_genome",
        ann_paths=["/home/yingzi/Visual-Instruction-Meta-Learning/workspace/lrv/filter_cap.json"],
        max_seq_length=256,
        sample=50000,
    ),
]

test_datasets = [
    dict(
        type="minigpt4",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/cc_sbu_align/image",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/cc_sbu_align/filter_cap.json",
        ],
        max_seq_length=512,
    ),
]

textvqa_datasets = [
    dict(
        type="textvqa",
        vis_root="/home/yingzi/MultiInstruct/TextVQA/train_val/train_images",
        ann_paths=[
            "/home/yingzi/MultiInstruct/TextVQA/train_val/TextVQA_0.5.1_train.json",
        ],
        max_seq_length=30,
        max_num_images=1,
    ),
]


vizwiz_datasets = [
    dict(
        type="vizwiz",
        vis_root="/home/yingzi/MultiInstruct/VizWiz/train",
        ann_paths=[
            "/home/yingzi/MultiInstruct/VizWiz/train.json",
        ],
        max_seq_length=30,
        max_num_images=1,
    ),
]

inst_octopus_mix = [
    dict(
        type="instruct",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=1024,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="octopus",
        retrieval_type="mix",
    ),
]


inst_otter_image = [
    dict(
        type="instruct",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=1024,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="otter",
        retrieval_type="image",
    ),
]

inst_otter_text = [
    dict(
        type="instruct",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=1024,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="otter",
        retrieval_type="text",
    ),
]

inst_octopus_random = [
    dict(
        type="instruct",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=1024,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="octopus",
        retrieval_type="random",
    ),
]

inst_octopus_mix_suffix = [
    dict(
        type="instruct",
        vis_root="/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/images",
        ann_paths=[
            "/home/yingzi/Visual-Instruction-Meta-Learning/workspace/viml/VIML-dataset/train",
        ],
        max_seq_length=1024,
        prompt_template="octopus",
        mode="train",
        with_exemplars=False,
        with_cot=False,
        suffix_loss=False,
        only_interleaved=True,
        interleaved_mode="octopus",
        retrieval_type="random",
    ),
]


DATASET_CONFIG = {
    "inst": inst_datasets,
    "inst_with_icl": inst_with_icl_datasets,
    "inst_with_cot": inst_with_cot_datasets,
    "inst_with_icl_cot": inst_with_icl_cot_datasets,
    "alignment": alignment_datasets,
    "test": test_datasets,
    "textvqa": textvqa_datasets,
    "vizwiz": vizwiz_datasets,
    "inst_octopus_mix": inst_octopus_mix,
    "inst_octopus_random": inst_octopus_random,
    "inst_otter_image": inst_otter_image,
    "inst_otter_text": inst_otter_text,
    "inst_octopus_mix_suffix": inst_octopus_mix_suffix
}