# 使用 LLaMA-Factory 微调 Spark2.5 1.7B / 4B

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是社区使用最广泛的微调框架。Spark2.5（`Spark2.5ForCausalLM`，混合 SWA+Full 注意力，逐头输出门控，`tie_word_embeddings=true`）通过 `trust_remote_code` 加载，在少量配置调整后即可适配 LLaMA-Factory 的标准训练流程。

本文档覆盖两条微调路径：

- **全参数 SFT + Muon 优化器** —— 对 2D 权重矩阵使用 Newton-Schulz 正交化，对 1D / embedding / lm_head 使用 AdamW。
- **LoRA SFT + Adam 优化器** —— 注入低秩适配器，冻结主干，运行默认 AdamW。**不要**在 LoRA 下启用 `use_muon`。

> **适用模型**：`spark2_5_1p7b`、`spark2_5_4b`（均为 `Spark2.5ForCausalLM`，`model_type=spark3`，通过 `trust_remote_code` 加载）。
>

---

## 安装

```bash
pip install "llamafactory==0.9.3"        # 或 `pip install llamafactory` 安装最新版
```

> **版本说明**：LLaMA-Factory 要求 `transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0`。4B 模型的 `config.json` 标注 `transformers_version: 4.57.1`，会触发版本校验失败 —— 设置 `DISABLE_VERSION_CHECK=1` 可跳过该检查。

---

## 1. 数据集准备

### 1.1 数据集格式

LLaMA-Factory 支持 **alpaca** 、 **sharegpt** 、**openai**三种格式。示例使用 openai格式：

```json
{"messages": [
  {"role": "system", "content": "you are a helpful assistant."},
  {"role": "user", "content": "用户问题"},
  {"role": "assistant", "content": "助手回答"}
]}
```

编辑 `data/dataset_info.json`，添加你的条目：

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

字段说明：
- `file_name`：数据文件名（放在 `data/` 下，或使用绝对路径）
- `formatting`：`sharegpt` 或 `alpaca` 或 `openai`
- `columns.messages`：数据格式中消息列表的字段名
- `tags`：role/content 标签字段名（可选；默认值与上方 schema 一致）

---

## 2. 全参数 SFT 配置（Muon 优化器）

Muon 将 2D 权重（排除 `embed`/`lm_head`）路由到 Newton-Schulz 正交化，其余参数（1D norm/bias、embedding、lm_head）路由到 AdamW。

### 2.1 1.7B配置示例：

```yaml
### model
model_name_or_path: model/spark2_5_1p7b
trust_remote_code: false

### method
stage: sft
do_train: true
finetuning_type: full                                # 全参数微调
use_muon: true                                       # 启用 Muon 优化器

### dataset
dataset: my_chat_data                                # dataset_info.json 中的键名
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
save_only_model: false                               # false：保存优化器状态以便恢复训练
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

### 2.2 4B配置示例：

4B 权重以 fp32 存储（约 15.3 GB），显存压力远高于 1.7B。请启用 gradient checkpointing 并优先使用多卡；

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
save_only_model: false                               # 恢复训练需要 Muon 的 momentum_buffer
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

## 3. LoRA SFT 配置（Adam 优化器）

LoRA 在目标线性层旁注入低秩 `A·B` 增量并冻结主干，可训练参数通常只有几 MB。优化器运行默认 AdamW（`adamw_torch`）—— **不要设置 `use_muon`**。

### 3.1 1.7B配置示例：

```yaml
### model
model_name_or_path: model/spark2_5_1p7b_configs   # 1.7B 模型目录的绝对路径
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
# 不要设置 use_muon —— 默认使用 AdamW

### lora
lora_rank: 8                                         # 8 / 16 / 32；越大越接近全参数
lora_alpha: 16                                       # 缩放系数，默认 = rank * 2
lora_dropout: 0.05
lora_target: all                                     # 所有线性层；或 q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj

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
per_device_train_batch_size: 2                       # LoRA 显存占用低，batch 可更大
gradient_accumulation_steps: 4                       # 有效 batch = 8
learning_rate: 5.0e-5                                # LoRA 学习率通常为 1e-5 ~ 1e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0
bf16: true
ddp_timeout: 180000000
```

### 3.2 4B配置示例：

4B 权重为 fp32（约 15.3 GB），但 LoRA 冻结了主干 —— 显存压力来自前向激活值而非优化器状态，因此单卡即可。建议保留 `gradient_checkpointing` 以削减激活显存。

```yaml
### model
model_name_or_path: models/spark2_5_4b    # 4B 模型目录的绝对路径
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
# 不要设置 use_muon —— 默认使用 AdamW

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
gradient_checkpointing: true                         # 4B 推荐：降低激活显存
ddp_timeout: 180000000
```


## 4. 训练

#### 4.1 多卡（8 卡）

LLaMA-Factory 会自动拉起 `torchrun` 进行分布式训练，1.7B 和 4B 均适用：

```bash
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISABLE_VERSION_CHECK=1

# 1.7B
llamafactory-cli train examples/torch_spark2_5/example_1p7b_sft_args.yaml
# 4B（推荐多卡 —— fp32 权重显存占用大）
llamafactory-cli train examples/torch_spark2_5/example_4b_sft_args.yaml
```

#### 4.2 后台运行（长时间训练）

```bash
# 1.7B
nohup llamafactory-cli train examples/torch_spark2_5/example_1p7b_sft_args.yaml \
  > saves/spark2_5_1p7b_sft/train.log 2>&1 &

# 4B
nohup llamafactory-cli train examples/torch_spark2_5/example_4b_sft_args.yaml \
  > saves/spark2_5_4b_sft/train.log 2>&1 &
```

#### 4.3 预期日志输出

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

看到 `Using Muon optimizer with N Muon params and M AdamW params.` 即确认 Muon 已生效。Muon 参数数量随层数线性增长：1.7B（28 层）约 168 个，4B（36 层）约 216 个（每层 6 个：`q_k_v_proj`、`g_proj`、`out_proj`、`gate_proj`、`up_proj`、`down_proj`）；AdamW 参数为 1D norm/bias。



## 5. 通过 WebUI 训练

LLaMA-Factory 自带 Gradio WebUI（`llamafactory-cli webui`），支持在浏览器中配置并启动训练。

### 5.1 启动 WebUI

```bash
export DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7860

llamafactory-cli webui
# 终端会打印：Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860
```

在浏览器中打开 `http://<host>:7860`。

### 5.2 通过 WebUI 进行全参数 SFT

WebUI 在 `_parse_train_args`（`src/llamafactory/webui/runner.py:133-298`）中构建参数字典，并以子进程方式拉起 `llamafactory-cli train <cmd.yaml>`。

1. **Model name**：自定义名称（如 `spark_1p7b`、`spark_4b`），用于组织 `saves/<name>/<finetuning_type>/`。
2. **Model path**：模型目录的绝对路径。
3. **Finetuning method**：`full`（Muon 不支持 lora/freeze）。
4. **Training stage**：`Supervised Fine-Tuning`。
5. **Template**：`deepseek3`。
6. **Dataset**：勾选已注册的数据集（如 `spark_sft`）。
7. 按 CLI 配置填写 **Train** 标签页。
8. **Preview command**：点击 "Preview" 查看生成的命令，然后点击 "Start"。

#### 5.2.1 附加参数（注入 Muon 与 gradient checkpointing）

WebUI 表单未暴露 `use_muon`、`warmup_ratio`、`gradient_checkpointing`、`save_only_model`。将它们以 JSON 形式填入 Train 标签页的 "Additional arguments" 输入框；通过 `args.update(json.loads(extra_args))`（`runner.py:187`）合并，会覆盖表单值。

**1.7B**：

```json
{"use_muon": true, "warmup_ratio": 0, "save_only_model": false, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000}
```

**4B**（追加 `gradient_checkpointing`）：

```json
{"use_muon": true, "warmup_ratio": 0, "save_only_model": false, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000, "gradient_checkpointing": true}
```

### 5.3 通过 WebUI 进行 LoRA SFT

LoRA SFT 与全参数 SFT 流程一致，但将微调方法切换为 `lora`，并暴露 LoRA 专属参数。优化器保持默认 AdamW —— **不要**注入 `use_muon`。

1. **Model name**：自定义名称（如 `spark_1p7b_lora`、`spark_4b_lora`），用于组织 `saves/<name>/lora/`。
2. **Model path**：模型目录的绝对路径。
3. **Finetuning method**：`lora`。
4. **Training stage**：`Supervised Fine-Tuning`。
5. **Template**：`spark`。
6. **Dataset**：勾选已注册的数据集（如 `spark_sft`）。
7. **LoRA 标签页**：设置 `lora_rank`（8 / 16 / 32）、`lora_alpha`（默认 = rank × 2）、`lora_dropout`（0.05）、`lora_target`（`all` 或逗号分隔的层名，如 `q_k_v_proj,g_proj,out_proj,gate_proj,up_proj,down_proj`）。
8. 按第 3 节的 CLI 配置填写 **Train** 标签页；LoRA 显存占用低，单卡通常足够。
9. **Preview command**：点击 "Preview" 查看生成的命令，然后点击 "Start"。

#### 5.3.1 附加参数（LoRA，AdamW —— 不使用 Muon）

LoRA 禁止启用 `use_muon`。4B 需追加 `gradient_checkpointing` 以削减激活显存。

**1.7B**：

```json
{"warmup_ratio": 0, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000}
```

**4B**（追加 `gradient_checkpointing`）：

```json
{"warmup_ratio": 0, "overwrite_cache": true, "dataloader_num_workers": 4, "ddp_timeout": 180000000, "gradient_checkpointing": true}
```

这与第 3 节的 CLI LoRA 配置相对应。


### 5.4 预览与监控

点击 **Preview** 后，输出框会显示等价的 CLI 命令（4B 示例）：

```
llamafactory-cli train /path/to/llamaboard_cache/cmd_<timestamp>.yaml
```

该临时 YAML 已包含合并后的参数 —— `use_muon: true`、`gradient_checkpointing: true`、`trust_remote_code: true` 等 —— 可下载核对，在启动前确认最终参数字典。

启动后：

- 右侧输出框会实时流式输出日志；进度条显示完成比例，下方绘制 loss 曲线。
- 子进程的 stdout/stderr 会写入 `<output_dir>/webui_subprocess.log` —— 可直接 `tail -f` 查看。
- 训练配置保存到 `<output_dir>/llamaboard_config.json`，便于复现。
- 点击 "Abort" 可中断训练（向子进程发送 SIGABRT）。
- **恢复训练**：在 Model 标签页的 "Checkpoint path" 中填入 `output_dir/checkpoint-{step}`（全参数 SFT）或适配器路径（LoRA），然后再次 Start（要求原始训练时 `save_only_model: false`）。

---

