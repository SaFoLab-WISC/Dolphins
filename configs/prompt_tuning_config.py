from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType
)

pt_config = dict(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=300,
    prompt_tuning_init_text="You are a multimodal assistant capable of performing various vision-language tasks. You will receive a number of different questions belonging to the same type of task and you are asked to answer them in a consistent way.\n",
    tokenizer_name_or_path="",
    token_dim=4096,
    num_transformer_submodules=1,
    num_attention_heads=32,
    num_layers=32,
)