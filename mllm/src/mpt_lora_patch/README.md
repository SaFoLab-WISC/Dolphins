# MPT-7B LoRA Patch

This is the Python model code for MPT-7B patched so that it can be used with a LoRA. Note that while I tested that it works and I get reasonable results out, it is very possible that the model isn't being trained correctly. The model code specifically says that left padding is not supported, but I forcibly did so and got decent results.

Note that when using LoRA, there is a strange quirk that prevents me from causing generation with an empty prompt.

I also included a model-agnostic `export_hf_checkpoint.py` script, which you can use to merge your lora back into a new full model. Once you do this, you do not need to use the patched version of the model code anymore. That being said, if you want to be able to load the model in 8bit you will still need it. The usage is `python export_hf_checkpoint.py <source> <lora> <dest>`.

If you would like to use this with `text-generation-webui`, apply the following patch:

```patch
--- a/modules/training.py
+++ b/modules/training.py
@@ -28,12 +28,13 @@ try:
     MODEL_CLASSES = {v: k for k, v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
 except:
     standard_modules = ["q_proj", "v_proj"]
-    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"]}
+    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"], "mpt": ["Wqkv"]}
     MODEL_CLASSES = {
         "LlamaForCausalLM": "llama",
         "OPTForCausalLM": "opt",
         "GPTJForCausalLM": "gptj",
-        "GPTNeoXForCausalLM": "gpt_neox"
+        "GPTNeoXForCausalLM": "gpt_neox",
+        "MPTForCausalLM": "mpt"
     }

 WANT_INTERRUPT = False
```

You will need to run the webui with these options:

```bash
python server.py --model mosaicml_mpt-7b-instruct --trust-remote-code --load-in-8bit
```

You may also need to patch `bitsandbytes/nn/modules.py` to prevent running out of VRAM when saving the LoRA:

```patch
--- a/modules.py
+++ b/modules.py
@@ -259,13 +259,13 @@
         if not self.state.has_fp16_weights and self.state.CB is None and self.state.CxB is not None:
             # reorder weight layout back from ampere/turing to row
             reorder_layout = True
-            weight_clone = self.weight.data.clone()
+            weight_clone = self.weight.data
         else:
             reorder_layout = False

         try:
             if reorder_layout:
-                self.weight.data = undo_layout(self.state.CxB, self.state.tile_indices)
+                self.weight.data = undo_layout(self.state.CxB.cpu(), self.state.tile_indices.cpu())

             super()._save_to_state_dict(destination, prefix, keep_vars)
```

(It resides in `miniconda3/envs/textgen/lib/python3.10/site-packages/bitsandbytes/nn/modules.py` for me.)

You can find the source model here: [mosaicml/mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)

The alterations are based on the [source code for the llama model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) from HF Transformers.

## Model License

CC-By-SA-3.0
