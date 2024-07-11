flamingov1_tuning_config = dict(
    target_modules=r'.*lang_encoder.*\.(q_proj|v_proj)',
    r=8 ,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save = ["embed_tokens", "lm_head", "perceiver", "gated_cross_attn_layer"]
)

mix_tuning_config = dict(
    target_modules=r'.*lang_encoder.*\.(q_proj|v_proj|to_q|to_kv)',
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save = ["embed_tokens"]
)


visual_tuning_config = dict(
    target_modules=r'transformer.blocks.*\.(Wqkv|out_proj)',
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # modules_to_save = ["embed_tokens", "lm_head", "gated_cross_attn_layer"]
)

# openflamingo_tuning_config = dict(
#     target_modules=r'transformer.blocks.*\.(Wqkv|out_proj|up_proj|down_proj)',
#     r=32,
#     lora_alpha=64,
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM",
#     # modules_to_save = ["embed_tokens", "lm_head", "gated_cross_attn_layer"]
# )

otter_tuning_config = dict(
    target_modules=["q_proj", "v_proj"],
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # modules_to_save = ["embed_tokens", "lm_head", "gated_cross_attn_layer"]
)

openflamingo_tuning_config = dict(
    target_modules=r'transformer.blocks.*\.(Wqkv|out_proj)',
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # modules_to_save = ["embed_tokens", "lm_head", "gated_cross_attn_layer"]
)

