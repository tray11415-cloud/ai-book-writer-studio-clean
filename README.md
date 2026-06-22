# AI Book Writer Studio

AI Book Writer Studio 是一套本機小說創作工作台，核心用途是把「設定、參考素材、章節分析、技法回灌、互動寫作、全文改寫」放在同一個 Gradio 介面裡操作。專案支援 OpenAI 相容 API、本機 LoRA 服務、Grok/xAI 分析路由，以及獨立的全文改寫 AGENT。

這份資料夾是可分享的乾淨副本：不包含 `.env`、API key、個人稿件、訓練素材、模型權重、生成輸出、虛擬環境、打包成品或版本紀錄。

## 功能總覽

- 互動式長篇小說寫作：依照故事設定、技法庫與章節方向生成正文。
- 參考書架：集中管理小說、報告、技法檔，供後續分析與寫作使用。
- 章節技法分析：把章節拆成技法、節奏、感官、人物動作與句式層次。
- 深度技法書庫：從參考書架建立可搜尋的技巧卡，並載入寫作 AGENT。
- 劇情編排：用多模型協作做主線、轉折、伏筆與章節節奏規劃。
- 技法回灌：把 `full_report.md` 蒸餾成精簡技法庫，再送回寫作欄位。
- 全文改寫 AGENT：支援 Grok 診斷、本機 LoRA、外部 API 或混合模式改寫。
- LoRA 訓練與服務：提供資料準備、訓練腳本、OpenAI 相容本機服務端。

## 需求

- Windows 10/11。
- Python 3.10 以上，建議 3.11。
- 可連線的 OpenAI 相容文字模型 API。
- 若要使用 Grok 分析，需 xAI API key。
- 若要使用本機 LoRA，需已下載的基礎模型與 LoRA 權重，並具備足夠 GPU/記憶體。

主要 Python 依賴列在：

- `requirements.txt`：Studio、Web app、代理與一般功能。
- `lora_training/requirements-lora.txt`：LoRA 訓練與本機推理服務。

## 快速啟動

1. 打開 PowerShell，進入專案資料夾：

```powershell
cd C:\Projects\ai-book-writer-clean-share
```

2. 複製公開環境範例：

```powershell
copy .env.example .env
```

3. 編輯 `.env`，填入自己的 API key 與模型設定。

4. 啟動完整 Studio：

```bat
start_studio.bat
```

第一次啟動會建立 `.venv` 並安裝依賴，時間較長。啟動後瀏覽器會開啟本機 Gradio 介面。

## 環境設定

`.env.example` 提供三條主要路由：

```env
# 寫作與改寫路由，預設走本機 compat proxy
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=nalang-xl-0826-10k
OPENAI_API_KEY=replace-with-your-nalang-api-key
UPSTREAM_OPENAI_BASE_URL=https://www.gpt4novel.com/api/xiaoshuoai/ext/v1

# 分析路由，供 Grok/xAI 使用
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-4.3
XAI_API_KEY=replace-with-your-xai-api-key

# 本機 LoRA 路由
UI_LORA_BASE_URL=http://127.0.0.1:8010/v1
UI_LORA_MODEL=qwen3-4b-novel-lora
```

注意事項：

- `.env` 不應提交或分享。
- API key 也可以在 Studio 的模型設定欄位輸入，並儲存到本機 `book_output/model_config.json`。
- 乾淨副本不包含 `book_output`，第一次執行時程式會自行建立。

## 啟動腳本

- `start_studio.bat`：啟動主要 Gradio Studio。
- `start_web.bat`：啟動簡化 Flask Web app。
- `start_lora_server.bat`：啟動 OpenAI 相容 LoRA 服務端。
- `prepare_lora_data.bat`：整理訓練語料。
- `train_lora_qwen.bat`：執行 LoRA 訓練。
- `改寫AGENT/start_rewrite_agent.bat`：啟動獨立全文改寫工具。
- `build_studio_launcher.bat`：重新打包 Windows 啟動器。乾淨副本未附帶 exe，需自行建置。

## Studio 工作流程

Studio 的分頁順序就是建議操作順序：

1. `設定 Core Settings`：設定模型、API、故事 Bible、Memory 與技法欄位。
2. `參考書架 Reference Library`：加入小說、報告、MD/TXT/JSON 參考來源。
3. `寫作 Interactive Writing`：依照設定與技法庫生成正文。
4. `改寫 Rewrite / Style Transfer`：做局部改寫、風格轉換與文字修整。
5. `章節技法分析 Chapter Craft Analysis`：逐章輸出 `full_report.md`。
6. `劇情編排 Plot Ideation`：規劃章節、伏筆、衝突與轉折。
7. `深度技法書庫 Deep Technique Book`：從參考書架建立可搜尋技法卡。
8. `存讀檔 Save / Load`：保存或載入專案狀態。
9. `技法回灌與檢閱 Skill / Technique Review`：蒸餾報告並載入寫作 AGENT。
10. `說明書 Manual`：在 app 裡閱讀完整面板操作說明。

更細的面板說明請看 `AI_Book_Writer_Studio_面板說明書.md`。

## 典型用法

### 從零開始寫一章

1. 到 `設定` 填入故事設定、角色、世界觀與模型路由。
2. 到 `參考書架` 加入想模仿或分析的素材。
3. 到 `深度技法書庫` 建立技法卡，搜尋需要的技法並載入寫作 AGENT。
4. 到 `寫作` 填入章節目標、場景、人物與禁忌事項。
5. 生成正文後，回到 `章節技法分析` 或 `改寫` 做修整。

### 把分析結果回灌成寫作技巧

1. 在 `章節技法分析` 產出 `book_output/chapter_craft_reports/<timestamp>/full_report.md`。
2. 到 `技法回灌與檢閱` 上傳或貼入 `full_report.md`。
3. 按 `Distill full_report To Compact Library`。
4. 按 `Load Distilled Library To Writing AGENT`。
5. 回到 `寫作`，下一次生成會讀到新的技法欄位。

### 使用參考書架建立深度技法書

1. 到 `參考書架` 加入多個 TXT/MD/JSON 來源。
2. 到 `深度技法書庫` 按 `Build Technique Book From Reference Shelf`。
3. 搜尋例如 `眼睛`、`手`、`動作`、`喝酒`、`節奏` 等關鍵詞。
4. 選中結果後載入寫作 AGENT。

## 全文改寫 AGENT

獨立工具位於 `改寫AGENT/`，適合處理整份 manuscript：

```powershell
cd C:\Projects\ai-book-writer-clean-share
.\改寫AGENT\start_rewrite_agent.bat
```

預設會開在 `http://127.0.0.1:7870` 起跳的可用 port。

主要流程：

1. 上傳 `.txt` 或 `.md`，或直接貼全文。
2. 可先用 Grok 做 `全篇分析診斷`。
3. 勾選 `套用最新診斷書`。
4. 選擇改寫模式：本機 LoRA、外部 API、混合模式或完整混合模式。
5. 執行後輸出到 `改寫AGENT/output/`。

詳細請看 `改寫AGENT/README.md`。

## LoRA 訓練與本機服務

乾淨副本只包含訓練與服務程式，不包含訓練素材、資料集、權重或 checkpoint。

建議流程：

1. 把你有權使用的素材放到 `lora_training/source_texts/`。
2. 執行：

```bat
prepare_lora_data.bat
```

3. 安裝 LoRA 依賴：

```powershell
.\.venv\Scripts\python.exe -m pip install -r lora_training\requirements-lora.txt
```

4. 執行訓練：

```bat
train_lora_qwen.bat
```

5. 啟動本機 OpenAI 相容服務：

```bat
start_lora_server.bat
```

LoRA 服務預設對外提供 `http://127.0.0.1:8010/v1`，Studio 會透過 `UI_LORA_BASE_URL` 與 `UI_LORA_MODEL` 連接。

## 目錄說明

```text
.
├── app_gradio.py                         # 主要 Gradio Studio
├── web_app.py                            # 簡化 Web app
├── compat_proxy.py                       # OpenAI 相容代理
├── config.py / env_utils.py              # 環境與模型設定
├── technique_library_builder.py          # 深度技法書庫
├── chapter_craft_skill.py                # 章節技法分析
├── report_technique_distiller.py         # full_report 蒸餾
├── lora_training/                        # LoRA 訓練與服務腳本
├── skills/                               # Studio 內部技能說明
├── 改寫AGENT/                            # 獨立全文改寫工具
├── AI_Book_Writer_Studio_面板說明書.md    # 面板操作手冊
├── .env.example                          # 可公開設定範本
└── requirements.txt                      # 主程式依賴
```

執行後會自動產生但不應分享的目錄：

- `.venv/`
- `book_output/`
- `改寫AGENT/output/`
- `lora_training/data/`
- `lora_training/runs/`
- `lora_training/lora_output/`

## 打包 Windows 啟動器

乾淨副本不包含已打包的 `StartAIBookWriterStudio.exe`。如需建立：

```bat
start_studio.bat
build_studio_launcher.bat
```

建置完成後會產生：

```text
dist/
StartAIBookWriterStudio.exe
```

分享前請確認 exe 沒有打包 `.env` 或個人資料。

## 常見問題

### 啟動時找不到 Python

安裝 Python，並勾選 `Add python.exe to PATH`。Windows 也可改用 `py -m venv .venv` 建立環境。

### Gradio 安裝失敗

先更新 pip：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 外部 API 回傳 502

先檢查 `.env` 裡的 `OPENAI_API_KEY` 或 `LLM_API_KEY` 是否存在，再檢查 `OPENAI_BASE_URL` 與 `UPSTREAM_OPENAI_BASE_URL`。

### Grok 分析不能用

檢查 `XAI_API_KEY`、`XAI_BASE_URL`、`XAI_MODEL`，也可以先在 Studio 的 `Test Analysis Grok` 測試。

### LoRA 服務不能連

先啟動 `start_lora_server.bat`，確認 `/health` 可回應，再回到 Studio 讀取 LoRA。

### 中文檔案亂碼

專案預設以 UTF-8/UTF-8-SIG 讀寫，舊檔若是 GB18030 或 Big5，建議先轉成 UTF-8。

## 分享前檢查

這份乾淨副本已移除：

- `.env` 與所有實際 API key。
- `.venv`、cache、`__pycache__`。
- `book_output`、改寫輸出、診斷報告與生成稿。
- 個人小說稿、訓練語料、訓練資料集。
- LoRA checkpoint、模型權重、訓練 runs。
- build/dist、exe、PyInstaller spec。
- 本機使用者路徑範例。

分享或上傳前仍建議再跑：

```powershell
rg -n --hidden "sk-[A-Za-z0-9_-]{20,}|xai-[A-Za-z0-9_-]{20,}|<your-user-name>|<private-project-name>" .
rg -n --hidden "OPENAI_API_KEY|XAI_API_KEY|LLM_API_KEY" .
```

只要結果沒有真實 key、個人路徑或私人內容，就可以交付。
