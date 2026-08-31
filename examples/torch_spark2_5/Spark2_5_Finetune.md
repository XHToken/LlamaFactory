# Fine-tune Spark2.5 1.7B / 4B with LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is the most-used community fine-tuning framework. Spark2.5 (`Spark2.5ForCausalLM`, hybrid SWA+Full attention, headwise output gate, `tie_word_embeddings=true`) is loaded via `trust_remote_code`, and works with LLaMA-Factory's standard recipe after a small config tweak.

Two fine-tuning paths are covered:

- **Full SFT + Muon optimizer** — Newton-Schulz orthogonalization on 2D weight matrices, AdamW on 1D / embedding / lm_head.
- **LoRA SFT + Adam optimizer** — inject low-rank adapters, freeze the backbone, run default AdamW. **Do not** enable `use_muon` under LoRA.

> **Applicable models**: `spark2_5_1p7b`, `spark2_5_4b` (both `Spark2.5ForCausalLM`, `model_type=spark3`, loaded with `trust_remote_code`).
>

---

## Install

```bash
pip install "llamafactory==0.9.3"        # or `pip install llamafactory` for latest
```

> **Version note**: LLaMA-Factory requires `transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0`. The 4B model's `config.json` labels `transformers_version: 4.57.1`, which trips the check — set `DISABLE_VERSION_CHECK=1` to bypass.

---

## 1. dataset prep

### 1.1 Dataset format

LLaMA‑Factory supports three formats: alpaca, sharegpt, and openai. The example uses the openai format.

```json
{"messages": [
  {"role": "system", "content": "you are a helpful assistant."},
  {"role": "user", "content": "user question"},
  {"role": "assistant", "content": "assistant answer"}
]}
```

Edit `data/dataset_info.json` and add your entry:

```json
{
  "my_chat_data": {
    "file_name": "my_chat_data.jsonl",
    "formatting": "openai",
    "columns": {
      "messages": "messages"
    },
  "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "tool",
      "function_tag": "function"
    }
  }
}
```

## 2. Full SFT config (Muon optimizer)

Muon routes 2D weights (excluding `embed`/`lm_head`) to Newton-Schulz orthogonalization and everything else (1D norms/bias, embeddings, lm_head) to AdamW.

### 2.1 Configuration example for the 1.7B‑parameter model：

```yaml
### model
model_name_or_path: model/spark2_5_1p7b  
trust_remote_code: false                            

### method
stage: sft
do_train: true
finetuning_type: full                                # full-parameter
use_muon: true                                       # enable Muon optimizer

### dataset
dataset: my_chat_data                                # key in dataset_info.json
template: spark                                  
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/spark2_5_1p7b_sft
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true
save_only_model: false                               # false: save optimizer state for resume
report_to: none

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8                    
learning_rate: 4.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr_rate: 8.0e-6
warmup_steps: 0
bf16: true
ddp_timeout: 180000000
```

### 2.2 Configuration example for the 4B‑parameter model：

4B weights are stored as fp32 (~15.3 GB); memory pressure is much higher than 1.7B. Enable gradient checkpointing and prefer multi-GPU; 

```yaml
### model
model_name_or_path: model/spark2_5_4b     
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
use_muon: true

### dataset
dataset: my_chat_data
template: spark
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/spark2_5_4b_sft
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true
save_only_model: false                               # Muon momentum_buffer needed for resume
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
weight_decay: 0.1
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8
learning_rate: 4.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
```


---

## 3. LoRA SFT config (Adam optimizer)

LoRA injects a low-rank `A·B` delta beside the target linear layers and freezes the backbone; trainable params are typically a few MB. The optimizer runs default AdamW (`adamw_torch`) — **do not set `use_muon`**.


### 3.1 Configuration example for the 1.7B‑parameter model：

```yaml
### model
model_name_or_path: model/spark2_5_1p7b_configs   # absolute path to 1.7B model dir
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
# do NOT set use_muon — defaults to AdamW

### lora
lora_rank: 8                                         # 8 / 16 / 32; larger = closer to full
lora_alpha: 16                                       # scale, default = rank * 2
lora_dropout: 0.05
lora_target: all                                     # all linear layers; or q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj

### dataset
dataset: my_chat_data
template: spark
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/spark2_5_1p7b_lora
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 2                       # LoRA is light, batch can be larger
gradient_accumulation_steps: 4                       # effective batch = 8
learning_rate: 5.0e-5                                # LoRA lr is typically 1e-5 ~ 1e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0
bf16: true
ddp_timeout: 180000000
```

### 3.2 Configuration example for the 4B‑parameter model：

4B weights are fp32 (~15.3 GB), but LoRA freezes the backbone — memory pressure comes from forward activations, not optimizer state, so a single GPU is enough. Keep `gradient_checkpointing` to cut activation memory.

```yaml
### model
model_name_or_path: models/spark2_5_4b    # absolute path to 4B model dir
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
# do NOT set use_muon — defaults to AdamW

### lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all

### dataset
dataset: my_chat_data
template: spark
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/spark2_5_4b_lora
logging_steps: 10
save_steps: 2000
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0
bf16: true
gradient_checkpointing: true                         # 4B recommended: lowers activation memory
ddp_timeout: 180000000
```


## 4. Train

#### 4.1 Multi-GPU (8 GPUs)

LLaMA-Factory auto-launches `torchrun` for distributed training; works for both 1.7B and 4B:

```bash
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISABLE_VERSION_CHECK=1

# 1.7B
llamafactory-cli train examples/torch_spark2_5/example_1p7b_sft_args.yaml
# 4B (multi-GPU recommended — fp32 weights are memory-heavy)
llamafactory-cli train examples/torch_spark2_5/example_4b_sft_args.yaml
```

#### 4.2 Background run (long training)

```bash
# 1.7B
nohup llamafactory-cli train examples/torch_spark2_5/example_1p7b_sft_args.yaml \
  > saves/spark2_5_1p7b_sft/train.log 2>&1 &

# 4B
nohup llamafactory-cli train examples/torch_spark2_5/example_1p7b_sft_args.yaml \
  > saves/spark2_5_4b_sft/train.log 2>&1 &
```

#### 4.3 Expected log output

```
[INFO] Process rank: 0, world size: 1, device: cuda:0, distributed training: False, compute dtype: torch.bfloat16
[INFO] loading configuration file .../config.json
[INFO] Loading dataset spark_sft.json...
Converting format of dataset (num_proc=16): 100%|██████████| 65949/65949
Running tokenizer on dataset (num_proc=16): 100%|██████████| 65949/65949
[INFO] Using Muon optimizer with 168 Muon params and 59 AdamW params.
{'loss': 0.8456, 'grad_norm': 1.234, 'learning_rate': 9.8e-06, 'epoch': 0.01}
{'loss': 0.8234, 'grad_norm': 1.123, 'learning_rate': 9.6e-06, 'epoch': 0.02}
...
```

Seeing `Using Muon optimizer with N Muon params and M AdamW params.` confirms Muon is active. Muon param count scales with layers: 1.7B (28 layers) ~168, 4B (36 layers) ~216 (6 per layer: `q_k_v_proj`, `g_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`); AdamW params are the 1D norms/bias.



## 5. Train via WebUI

LLaMA-Factory ships a Gradio WebUI (`llamafactory-cli webui`) for browser-based config and launch.

### 5.1 Launch the WebUI

```bash
export DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860

llamafactory-cli webui
# terminal prints: Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860
```

Open `http://<host>:7860` in a browser.

### 5.2 Full SFT via WebUI

The WebUI builds the arg dict in `_parse_train_args` (`src/llamafactory/webui/runner.py:133-298`) and spawns `llamafactory-cli train <cmd.yaml>` as a subprocess.

1. **Model name**: a custom name (e.g. `spark_1p7b`, `spark_4b`) used to organize `saves/<name>/<finetuning_type>/`.
2. **Model path**: absolute path to the model directory.
3. **Finetuning method**: `full` (Muon does not support lora/freeze).
4. **Training stage**: `Supervised Fine-Tuning`.
5. **Template**: `deepseek3`.
6. **Dataset**: tick the registered dataset (e.g. `spark_sft`).
7. Fill the **Train** tab according to CLI . 
8. **Preview command**: click "Preview" to inspect the generated command, then "Start".

#### 5.2.1 Additional arguments (inject Muon & gradient checkpointing)

The WebUI form doesn't expose `use_muon`, `warmup_ratio`, `gradient_checkpointing`, `save_only_model`. Put them in the Train tab's "Additional arguments" box as JSON; it is merged via `args.update(json.loads(extra_args))` (`runner.py:187`), overriding form values.

**1.7B**:

```json
{"use_muon": true, "warmup_ratio": 0, "save_only_model": false, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000}
```

**4B** (add `gradient_checkpointing`):

```json
{"use_muon": true, "warmup_ratio": 0, "save_only_model": false, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000, "gradient_checkpointing": true}
```
### 5.3 LoRA SFT via WebUI

LoRA SFT mirrors Full SFT but switches the finetuning method to `lora` and exposes LoRA-specific parameters. The optimizer stays default AdamW — **do not** inject `use_muon`.

1. **Model name**: a custom name (e.g. `spark_1p7b_lora`, `spark_4b_lora`) used to organize `saves/<name>/lora/`.
2. **Model path**: absolute path to the model directory.
3. **Finetuning method**: `lora`.
4. **Training stage**: `Supervised Fine-Tuning`.
5. **Template**: `deepseek3`.
6. **Dataset**: tick the registered dataset (e.g. `spark_sft`).
7. **LoRA tab**: set `lora_rank` (8 / 16 / 32), `lora_alpha` (default = rank × 2), `lora_dropout` (0.05), `lora_target` (`all` or comma-separated layer names such as `q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj`).
8. Fill the **Train** tab per the CLI configs in section 3; LoRA is memory-light, so a single GPU is usually enough.
9. **Preview command**: click "Preview" to inspect the generated command, then "Start".

#### 5.3.1 Additional arguments (LoRA, AdamW — no Muon)

LoRA must NOT enable `use_muon`. For 4B, add `gradient_checkpointing` to cut activation memory.

**1.7B**:

```json
{"warmup_ratio": 0, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000}
```

**4B** (add `gradient_checkpointing`):

```json
{"warmup_ratio": 0, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000, "gradient_checkpointing": true}
```

This parallels the CLI LoRA configs in section 3.


### 5.4 Preview & monitor

After clicking **Preview**, the output box shows the equivalent CLI command (4B example):

```
llamafactory-cli train /path/to/llamaboard_cache/cmd_<timestamp>.yaml
```

This temporary YAML already contains the merged parameters — `use_muon: true`, `gradient_checkpointing: true`, `trust_remote_code: true`, etc. — which you can download and inspect to verify the final arg dict before launch.

Once started:

- The right-hand output box streams logs live; the progress bar shows completion ratio and the loss curve draws below.
- The subprocess stdout/stderr is written to `<output_dir>/webui_subprocess.log` — `tail -f` it directly.
- The training config is saved to `<output_dir>/llamaboard_config.json` for reproducibility.
- Click "Abort" to interrupt training (sends SIGABRT to the subprocess).
- **Resume**: in the Model tab's "Checkpoint path", fill `output_dir/checkpoint-{step}` (full SFT) or the adapter path (LoRA), then Start again (requires `save_only_model: false` during the original run).

---


