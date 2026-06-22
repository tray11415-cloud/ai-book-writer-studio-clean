# LoRA training notes

This folder now has a local Qwen LoRA workflow for novel-writing experiments.

## Plain workflow

1. Put real novel or style-reference files anywhere in this project folder, preferably under `lora_training/source_texts`.
   Supported source formats include `.txt`, `.md`, `.json`, `.jsonl`, `.csv`, `.yaml`, and `.epub`.
2. Run `prepare_lora_data.bat`.
3. Check `lora_training/data/dataset_report.json`.
4. If the report has enough samples, run `train_lora_qwen.bat`.
5. The adapter will be saved to `lora_training/lora_output/qwen3_4b_novel_lora`.

If `lora_training/data/train.jsonl` already exists, `train_lora_qwen.bat` will use it directly instead of rebuilding the dataset.

## Use the LoRA in the writing app

Run `start_studio.bat`, then open **Core Settings** and choose **Local Qwen LoRA** from **Provider Preset**.

The app will call:

- Base URL: `http://127.0.0.1:8010/v1`
- Model: `qwen3-4b-novel-lora`

The LoRA server starts automatically when the preset is used. You can also start it manually with `start_lora_server.bat`.

The first LoRA generation can take a while because the Qwen3-4B base model and adapter need to load into VRAM. After that, later generations reuse the loaded model.

## Hybrid API + LoRA writing

For a mixed workflow, choose **Hybrid DZMM + Local LoRA** from **Provider Preset**.

That mode does three passes:

1. The external API plans the next scene and checks continuity.
2. The local Qwen LoRA writes the prose draft in the trained style.
3. The external API polishes the draft for coherence and flow.

This keeps the external API useful for structure while letting the local LoRA carry the trained writing texture.

## Important

The current project folder mostly contains Python source code. That is not good training data for a novel-writing LoRA.
For a useful adapter, add real prose first. A practical minimum is dozens of long passages; hundreds are better.

## 16GB VRAM default

`train_lora_qwen.bat` defaults to:

- Base model: `Qwen/Qwen3-4B`
- 4-bit QLoRA
- Batch size: 1
- Gradient accumulation: 8
- Max length: 2048

This is intentionally conservative for a 16GB GPU. After a successful run, you can try `Qwen/Qwen3-8B`, but it may need a smaller `--max-length`.

The local training environment lives in `lora_training/.venv` and installs the CUDA 12.8 PyTorch wheel for RTX 50-series GPUs.

## Encoding cleanup

`prepare_lora_data.bat` tries to decode text as UTF-8, GB18030/GBK, Big5/CP950, then converts Simplified Chinese to Taiwanese Traditional Chinese with OpenCC when available.

It writes:

- `lora_training/data/clean_corpus.txt`
- `lora_training/data/train.jsonl`
- `lora_training/data/dataset_report.json`
