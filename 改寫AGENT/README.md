# 改寫AGENT

改寫AGENT 是獨立的全文改寫工具，適合處理長篇 `.txt` 或 `.md`。它會把全文切成多段，逐段改寫，再維護故事脈絡、主敘事骨幹、章節接續要點與重複檢查。

## 功能

- 支援全文上傳或直接貼文。
- 可先用 Grok 做全篇診斷，再把診斷結果套用到改寫。
- 支援四種模式：本機 LoRA、外部 API、LoRA 改寫加 API 校稿、API 規劃加 LoRA 改寫加 API 校稿。
- 支援改寫、增補擴寫、精簡刪減。
- 會輸出正文、JSON 紀錄與 QA 報告。
- 會自動清理模型輸出的 prompt 殘留、重複段落與斷尾。

## 啟動

從專案根目錄執行：

```powershell
cd C:\Projects\ai-book-writer-clean-share
.\改寫AGENT\start_rewrite_agent.bat
```

啟動後會開在 `http://127.0.0.1:7870` 起跳的可用 port。

## 基本流程

1. 上傳 `.txt` 或 `.md`，或直接貼上全文。
2. 填寫改寫要求，例如文風、保留設定、不可改動的人物關係。
3. 可加入參考風格檔或風格片段。
4. 如需全篇一致性，先執行 `全篇分析診斷（Grok）`。
5. 勾選 `套用最新診斷書`。
6. 選擇模式、強度與操作。
7. 按 `開始全篇改寫`。

## 模式說明

- `本地 LoRA`：只用本機 LoRA，速度與品質取決於本機模型。
- `外部 API`：只用外部 API，適合穩定校稿或高品質改寫。
- `混合：LoRA改寫 -> API校稿`：先用 LoRA 生成草稿，再用外部 API 修整。
- `完整混合：API規劃 -> LoRA改寫 -> API校稿`：先由 API 規劃段落，再由 LoRA 改寫，最後 API 校稿。

## 操作模式

- `改寫`：保留劇情，重寫文風與表達。
- `增補擴寫`：增加場景、心理、動作與細節。
- `精簡刪減`：保留主線，壓縮冗詞與重複內容。

## API 設定

改寫AGENT 會讀取專案根目錄的 `.env`。常用設定：

```env
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=nalang-xl-0826-10k
OPENAI_API_KEY=replace-with-your-api-key

XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4.3
XAI_API_KEY=replace-with-your-xai-api-key
```

若外部 API 需要本機代理，請確認 `compat_proxy.py` 可啟動，並確認上游 `UPSTREAM_OPENAI_BASE_URL` 設定正確。

## 輸出

改寫結果會寫到：

```text
改寫AGENT/output/
```

常見檔案：

- `rewrite_<timestamp>.txt`：最終改寫正文。
- `rewrite_<timestamp>.json`：每段設定、輸出與處理紀錄。
- `rewrite_<timestamp>_qa.txt`：重複、斷尾、清理結果等 QA 報告。
- `改寫診斷書_<timestamp>.md`：Grok 產出的可讀診斷書。
- `改寫診斷書_<timestamp>.json`：可被改寫流程套用的診斷資料。

## CLI 用法

直接改寫：

```powershell
.\.venv\Scripts\python.exe .\改寫AGENT\rewrite_agent.py .\input\source.txt `
  --operation 改寫 `
  --strength 強改寫 `
  --instruction "保留劇情，改成更成熟的繁體中文小說文風。"
```

先分析再改寫：

```powershell
.\.venv\Scripts\python.exe .\改寫AGENT\analysis_agent.py .\input\source.txt

.\.venv\Scripts\python.exe .\改寫AGENT\rewrite_agent.py .\input\source.txt `
  --diagnosis ".\改寫AGENT\output\改寫診斷書_<timestamp>.json" `
  --operation 改寫
```

常用參數：

- `--mode`：選擇本機 LoRA、外部 API 或混合模式。
- `--operation`：`改寫`、`增補擴寫`、`精簡刪減`。
- `--chunk-chars`：每段原文字數。
- `--max-tokens`：每段輸出 token 上限。
- `--notes`：全篇固定設定與連貫要求。
- `--style-file`：參考風格檔，可重複使用。
- `--no-story-context`：關閉跨段落故事摘要。
- `--no-chapter-continuity`：關閉主敘事模組與章節接續。
- `--dedup-similarity`：近似重複段落判定門檻。

## 分享注意

`output/`、`input/`、診斷書、改寫結果、API key 與原稿都不應放進公開副本。這份乾淨副本已移除既有輸出與個人路徑，只保留工具程式。

