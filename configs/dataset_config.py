
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



DATASET_CONFIG = {
    "alignment": alignment_datasets,
}
