# 分享副本清理紀錄

清理目標：建立可分享、可重新安裝、不可直接還原私人工作內容的專案副本。

副本位置：

```text
<share-folder>
```

## 已排除內容

- `.env` 與實際 API key。
- `.git`、版本紀錄與本機 IDE/cache 資料。
- `.venv`、`__pycache__`、測試與 Python 快取。
- `build/`、`dist/`、exe、PyInstaller spec。
- `book_output/` 內的生成稿、分析報告、技法庫與模型設定。
- 根目錄私人 `.txt`、`.json` 稿件與設定草稿。
- `改寫AGENT/output/` 與既有進度快照。
- `lora_training/source_texts/` 的訓練原文。
- `lora_training/data/` 的訓練資料集。
- `lora_training/runs/` 與 `lora_training/lora_output/` 的訓練輸出、checkpoint、權重。
- 壓縮檔、資料庫、log、模型權重與大型二進位檔。

## 已保留內容

- Python 原始碼。
- 啟動與訓練腳本。
- `requirements.txt` 與 LoRA 依賴清單。
- `.env.example` 公開環境範本。
- Studio 面板說明書。
- 改寫AGENT 原始碼與說明書。
- Skill 說明與 prompt contract。

## 已修正痕跡

- 把 UI 和文件中的本機使用者路徑範例改成 `C:\path\to\...`。
- 重寫主 README，改成可交付說明書。
- 重寫改寫AGENT README，移除亂碼與本機路徑。
- 新增訓練素材資料夾的占位說明。

## 交付前建議再查一次

```powershell
cd <share-folder>
rg -n --hidden "sk-[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9_-]{20,}|<your-user-name>|<private-project-name>" .
rg -n --hidden "OPENAI_API_KEY|XAI_API_KEY|LLM_API_KEY" .
```

允許出現 `.env.example` 的 placeholder，但不應出現真實 key、私人稿件或個人路徑。
