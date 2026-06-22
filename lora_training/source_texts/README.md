# LoRA 訓練素材放置處

這份乾淨副本不包含任何訓練原文或私人素材。

使用者可把自己有權使用的 `.txt`、`.md` 或其他可被 `prepare_corpus.py` 處理的文本放到這個資料夾，再執行：

```bat
prepare_lora_data.bat
```

產生的訓練資料會寫入 `lora_training/data/`，訓練輸出會寫入 `lora_training/runs/` 或 `lora_training/lora_output/`。這些目錄都不建議分享。

