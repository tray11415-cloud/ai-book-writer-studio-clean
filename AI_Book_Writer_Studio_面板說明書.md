# AI Book Writer Studio 面板說明書

本說明書介紹 `StartAIBookWriterStudio.exe` 啟動後的 Gradio Studio。這個版本把整個流程重新整理成一條**連貫、統整、由上而下**的工作流：

> **設定 → 蒐集參考 → 深度技法分析 → 灌進寫作腦 → 寫作 → 打磨**

分頁順序就是建議的操作順序。你也可以打開介面內的 **`10. 說明書 Manual`** 分頁，直接在 App 裡讀這份文件。

---

## 啟動方式

1. 在專案根目錄雙擊 `StartAIBookWriterStudio.exe`。
2. 啟動器會自動選擇可用的本機 port（通常是 `7860` 或附近）。
3. 瀏覽器開啟後即可使用 Studio。

模型路由原則：

- **寫作**：NALANG + Local LoRA。
- **分析**：Grok。
- **劇情規劃**：Grok + NALANG + Local LoRA。
- **深度技法書庫、Technique Finder、full_report 蒸餾、Story Skill 網路探索**：主要使用 Grok；`Dry Run` 可本地測試不花 API。

---

## 統整工作流總覽

| 階段 | 分頁 | 你在這裡做什麼 |
|---|---|---|
| ① 設定 | `1. 設定 Core Settings` | 設好寫作／分析／LoRA 模型，填世界觀、角色、記憶 |
| ② 蒐集參考 | `2. 參考書架 Reference Library` | **隨時**上傳或刪除小說、報告當參考；全域共用 |
| ③ 深度分析 | `5. 章節技法分析` ＋ `7. 深度技法書庫` | 把參考拆成「眼睛怎麼寫、喝酒怎麼寫」級別的技法卡 |
| ④ 灌進寫作腦 | `7. 深度技法書庫` 搜尋載入 ／ `9. 技法回灌與檢閱` ／ `11. 故事技能` | 把技法卡或 story skill 載入寫作 AGENT |
| ⑤ 寫作 | `3. 寫作 Interactive Writing` | 用載入的技法生成續寫 |
| ⑥ 打磨 | `4. 改寫`、`6. 劇情編排`、`9. 檢閱` | 改寫風格、編排劇情、檢閱技法品質 |

兩個關鍵設計：

1. **參考書架是全域的。** 在 `2. 參考書架` 加入或刪減的小說參考，會自動供 `7. 深度技法書庫` 建庫使用，不必每次重新上傳。書架隨時可增刪。
2. **分析不再淺淺幾句帶過。** 深度技法書庫的每張卡，都會拆到**部位／動作如何一句一句寫**的層級（見後面〈深度技法分析原理〉）。

---

## 1. 設定 Core Settings

全域設定與寫作 AGENT 的參考資料區。大部分面板的產出，最後都會回到這裡成為寫作參考。

### Writing Provider Preset

選擇寫作模型路由。

- `NALANG + Local LoRA`：建議預設。NALANG 做規劃與潤飾，LoRA 做本地風格草稿。
- `Current Proxy`、`OpenAI Compatible`、`Local Qwen LoRA`：用於不同後端或測試。

### Writing API Key / Base URL / Model / Pipeline

設定寫作模型與流程（單模型或 Hybrid）。

### Analysis / Grok Routing

設定分析模型，供 Style DNA、Story Chronicle、章節技法分析、劇情編排、深度技法書庫與 full_report 蒸餾使用。可用 `Test Analysis Grok` 測連線。

### LoRA Routing

設定本地 LoRA 寫作路由，可用 `Test LoRA` 測連線。

### Save / Load Model Config

`Save Model Config` 與 `Load Model Config` 會把以上模型設定保存在 `book_output/model_config.json`。

### World / Story Background、Story Memory

- `World / Story Background`：世界觀、故事基礎設定、主要地點、核心矛盾。
- `Story Memory`：必須長期記住的劇情狀態（角色關係、已發生事件、伏筆、禁忌、當前任務）。full_report 蒸餾或技法載入時，也可把簡短提醒追加到這裡。

### Characters / Lorebook

- `Characters`：角色表（Name／Role／Traits）。
- `Lorebook`：術語、地點、宗門、物品、制度等設定表（Keyword／Content）。

### Style DNA、Story Chronicle

- `Style DNA`：上傳 `.txt` 樣本 → `Analyze Style DNA` → 產出 `Style DNA` 與 `Few-Shot Samples`。
- `Story Chronicle`：上傳故事 `.txt` → `Build Story Chronicle` → 產出年表、角色狀態、未解伏筆與後續方向。

### Technique Library

**寫作 AGENT 真正會讀取的技法參考欄位。** `7. 深度技法書庫` 的搜尋載入、`9. 檢閱` 的 full_report 蒸餾，都會把技法寫進這裡。

建議放精簡後的抽象技法模板，不要貼整份報告，例如：

- 用具體動作外化情緒。
- 每場戲安排目標、阻力、反應、轉折。
- 段尾保留未解問題或行動鉤子。

---

## 2. 參考書架 Reference Library（全域，隨時新增／刪減）

**這是整個 Studio 的參考來源中樞。** 把多本小說 TXT、`full_report.md`、Technique Finder JSON 放進這個書架；`7. 深度技法書庫` 會直接從這裡建立深度技法卡，不必每次重新上傳。

你**不需要一次準備好所有小說**——隨時可以回來新增、刪除、清空或重新載入。

### 輸入

- `Add Source Files (.md / .txt / .json)`：一次上傳多個檔案。
- `Add File / Folder Paths`：貼單一檔案、多個路徑，或資料夾路徑（資料夾會自動讀取裡面的 MD/TXT/JSON）。
- `Pasted Source Label` / `Pasted Source Text`：替臨時貼上的片段命名並貼上內容。
- `Max Chars / Source`：每個來源最多讀多少字。

### 按鈕

- `Add To Reference Shelf`：把目前上傳、路徑或貼上的內容加入書架。
- `Load Saved Shelf`：重新載入保存的 `book_output/technique_reference_shelf.json`。
- `Remove Selected References`：在 `Current References` 選取後刪除。
- `Clear Shelf`：清空目前書架。

整理好之後，到 `7. 深度技法書庫` 按 **Build Technique Book From Reference Shelf** 開始深度分析。

保存到：`book_output/technique_reference_shelf.json`

---

## 3. 寫作 Interactive Writing

正式寫作面板。

### Full Story / Undo / Clear

- `Full Story`：目前全文或目前章節正文。生成新內容時，模型會讀取最近上下文並追加。
- `Undo`：撤回上一次生成。`Clear Story`：清空全文。

### Story Style / Custom Style

選擇或自訂故事風格：`Standard`、`Cinematic`、`Slow Burn`、`Dark`、`Romance`、`Suspense`、`Custom`。

### 取樣參數

- `Temperature`／`Top-P`：穩定就調低 Temperature，要變化就調高。
- `Frequency Penalty`：避免重複句式時調高。
- `Presence Penalty`、`Max Tokens`：控制探索度與單次長度。（主寫作面板的 token 上限與獨立改寫工具分開。）

### Global & Advanced

- **Art & Texture**：`Linguistic Texture`、`Narrative Pacing`、`Intensity`、`Point of View`。
- **Sensory Weights**：視覺、聽覺、嗅覺、觸覺、味覺權重。
- **Format & Directing**：`Output Language`、`Paragraph Density`、`Dialogue Ratio`、`Focus Words`、`Avoid Words`、`Director Note Override`、`Context Window Hint`。

### Director Instruction（最高優先級）

本次生成的直接命令，優先級高於 Technique Library、Style DNA、Story Memory 等低優先級參考。範例：

```text
寫下一段雨夜入城。主角不要直接說出恐懼，而是用動作與環境聲表現。他在城門口遇到一名不該出現的人，段尾留下鉤子。
```

### Generate Continuation

開始續寫。Hybrid 模式流程：NALANG 規劃場景 → Local LoRA 寫草稿 → NALANG 潤飾整合。

> 本版輸入結構以 **Story Instruction 為主導**：故事指令會放在 prompt 的最前與最後，背景/角色/技法庫等都被降為低優先級參考，衝突時讓位給故事指令。

### 📋 相關技法模板（遇到主題自動檢視該怎麼寫）

進行中的文章或故事指令一旦出現特定描寫主題（眼睛、喝酒、拔劍、壓抑、章尾鉤子…），這個面板會**自動跳出**對應的「該怎麼寫」技法模板，讓你即時對照寫法。

- **自動偵測**（預設開啟）：每次生成完、或修改 Story Instruction 時，自動更新模板。
- **最多主題數**：一次最多顯示幾個命中主題。
- **立即檢視相關技法模板**：手動觸發一次偵測。
- **套用到寫作參考**：把目前命中的技法模板一鍵寫進 `Technique Library`，讓下次生成直接參考。

偵測來源優先順序：目前 session 建好的深度技法書庫 → 最近一次保存的書庫 → 內建技法 taxonomy。所以**沒有建過書庫也能用**；先在 `7. 深度技法書庫` 建一次，模板會更貼合你的參考小說。

每張模板會精簡顯示：觸發詞、一句話、**怎麼寫（部位／動作解剖）**、句法節奏、用詞、感官、弱寫→強寫、避免事項與導演指令。

---

## 4. 改寫 Rewrite / Style Transfer

- `Style Reference Files`：上傳 `.txt` 風格參考。
- `Rewrite Instruction`：輸入改寫方向（更冷峻、更詩性、更像電影分鏡、保留情節但增加壓迫感…）。
- `Target Draft` / `Target Length`：要改寫的草稿與目標長度。
- `Rewrite`：執行，結果出現在 `Rewrite Output`。

---

## 5. 章節技法分析 Chapter Craft Analysis

逐章分析小說寫作技巧，產出 `full_report.md`。這份報告之後可在 `9. 檢閱` 蒸餾成精簡技法庫回灌寫作，也可加進 `2. 參考書架` 當深度技法書庫的來源。

### 輸入來源（三選一）

- `Novel TXT`：上傳整本 TXT。
- `Pasted Novel Text`：直接貼正文。
- `Chapter Directory URL`：靜態章節目錄網址。

### 參數

- `/goal`：分析目的（預設分析章節功能、敘事節奏、人物推進、衝突設計、語言風格、可複製技法、可改善處）。
- `Chapter Limit (0 = Full Book)`：`0` 全本、`1` 只第一章、`10` 前十章。
- `Max Chars / Chapter / Auto Chunk`：每章送入模型的字數；TXT 沒有章節標題時，會依此自動切成 `Auto Chunk 001`、`002`…
- `Dry Run`：本地流程測試，不呼叫 API。

### 本版深化

合成報告新增 **「逐句寫法解剖」** 區段：會挑本章 2–3 個關鍵片段（某個身體部位、某個動作、某段對話），拆解它「一句一句怎麼寫成」，並附弱寫→強寫對照。

報告保存到：`book_output/chapter_craft_reports/<timestamp>/full_report.md`

---

## 6. 劇情編排 Plot Ideation

- `Story Seed / Premise`：故事種子（主角、世界、核心矛盾、想要的走向）。
- `Reference Text / TXT / URL`：可選參考文本，會先蒸餾技法 DNA 再用於編排，不會直接續寫原作。
- `Genre / Tone`、`Arrangement Mode`、`Target Chapters`、`Reference Chapter Limit`：類型語氣、編排模式、章數與參考章數。
- `Generate Plot Ideation`：產出核心賣點、劇情引擎、章節表、場景清單、衝突設計，以及可貼進寫作面板的 Director Instruction。

保存到：`book_output/plot_ideation/<timestamp>/plot_ideation.md`

---

## 7. 深度技法書庫 Deep Technique Book（核心）

把 `2. 參考書架` 的參考，蒸餾成**可搜尋、階層式、深度**的技法卡書庫，再載入寫作 AGENT。這是「**眼睛怎麼寫、喝酒怎麼寫、拔劍怎麼寫**」的主工作流。

書庫以大分類／小分類存放，例如：

- `容顏描寫 / 眼睛描寫`、`容顏描寫 / 嘴巴描寫`
- `身材描寫 / 手與指節描寫`、`身材描寫 / 腰身姿態描寫`
- `動作描寫 / 喝酒飲茶描寫`、`動作描寫 / 拔劍出招描寫`、`動作描寫 / 走路步伐描寫`
- `情緒描寫 / 壓抑情緒描寫`、`場景氛圍 / 夜色燭火描寫`、`節奏章法 / 章尾轉折`

### Build Deep Technique Book From Shelf

- `/goal`：指定要蒸餾的技法方向。
- `Output Language`：輸出語言。
- `Max Entries / Subcategory`：每個小分類最多保留幾張卡。
- `Dry Run / Local Categorizer`：本地分類測試，不花 Grok API；關掉後用 Grok 正式蒸餾。

產出 `Technique Book Markdown`、`Technique Book JSON` 與前端預覽，保存到 `book_output/integrated_technique_libraries/<timestamp>/`。

### Search Technique Book / Load To Writing AGENT

- `Search Query`：輸入 `眼睛`、`嘴巴`、`手`、`腰身`、`喝酒`、`拔劍`、`壓抑`、`章尾鉤子` 等。
- `Category / Subcategory`：直接選大／小分類。
- `Existing Library JSON Path`：可指定之前的 JSON；留空會用目前 session 或最近一次保存的書庫。
- `Search Technique Book`：只查詢。
- `Load Search Results To Writing AGENT`：把結果寫入 `Technique Library`，並可追加 Story Memory／Director Instruction。
- `Load Latest Book Library To Writing AGENT`：不查詢，直接載入最新書庫前幾張卡。

### Legacy / Focused 工具（折疊區）

- `Distill One Novel To Technique Finder`：單本小說 → Technique Finder 卡片庫（`book_output/scene_techniques/library/<timestamp>/`）。
- `Focused Technique Sheet`：針對一個具體 `Specific Scene / Action / Situation / Desired Effect` 整理深度手法（已深化，見下節）。

---

## 深度技法分析原理 —「如何寫身體部位／如何寫動作」

這是本版最重要的升級，回應「分析不要那麼淺淺幾句帶過」的需求。

過去技法卡只有泛泛的 5 個欄位（Deep Breakdown、Detail Lenses、Micro Techniques、Common Mistakes、Practice Prompts），常常只是「要生動描寫」這種空話。本版為每張卡新增 **5 個深度欄位**，強制分析拆到**一句一句怎麼寫**的層級：

| 深度欄位 | 它回答什麼 | 例（眼睛描寫） |
|---|---|---|
| **Anatomy Breakdown 部位／動作解剖** | 把這個部位／動作拆成「可觀察的微單元」並排出書寫順序 | 視線落點 → 眼神變化 → 停留時長 → 一個光濕細節 → 對方反應 |
| **Sentence Rhythm 句法節奏** | 句子長短、停頓、標點怎麼配合 | 落點句短、動作句更短；注視的停留用句號懸住 |
| **Word Palette 用詞調色盤** | 該用哪些動詞／名詞／質感詞，該避免哪些抽象詞 | 用：抬、垂、斜、瞇；避免：深邃迷人、水汪汪 |
| **Sensory Layering 感官分層** | 哪個感官領頭、其餘的先後與份量 | 視覺主導（落點＋一個光濕細節），其餘交給對方反應 |
| **Weak vs Strong 弱寫→強寫** | 一組短句級對照，看出技法差別 | 弱：他有一雙深邃的眼睛。<br>強：他的目光落在她攥緊的手上，停了一瞬，才慢慢抬到她臉上。 |

### 兩個實際例子

**寫眼睛（容顏描寫 / 眼睛描寫）**

- 解剖順序：眼睛先看向哪裡（落點本身要洩露情緒）→ 抬眼／垂眸／斜睨擇一 → 用「半息／一瞬／久久」量化注視時長 → 只給一個眼部濕度或光的細節 → 讓被看的人被迫反應。
- 句法：落點句短，注視的停留用句號懸住。
- 用詞：抬、垂、斜、掃、定、瞇；避免「美麗動人、深邃迷人」。

**寫喝酒（動作描寫 / 喝酒飲茶描寫）**

- 解剖順序：端杯前先給目的（拖延、試探、壓住怒意）→ 器物聲做節拍（杯底落案、酒液入盞）→ 入口寫停頓、喉結、餘味，而非只寫「喝下去」→ 放杯的位置與輕重是這一回合的句點 → 讓對方據此判斷，喝酒成攻防。
- 句法：倒、舉、抿、放用短句一拍一拍推進，像慢動作。
- 弱寫→強寫：
  - 弱：他端起酒杯一飲而盡，然後說道。
  - 強：他指尖在杯沿頓了頓，才抿了一口，慢慢嚥下。酒盞擱回案上，發出極輕的一聲。「所以呢？」

### 這些深度從哪來

- **正式（Grok）路徑**：建庫時關掉 `Dry Run`，會用一份強化過的 prompt——附深度規格、每欄位的硬性要求、以及一張合格卡片範例——驅動 Grok 從你的參考小說裡，抽象出上述五個深度欄位。
- **本地（Dry Run）路徑**：不花 API 也能看到深度。系統內建了針對眼睛、嘴巴、臉部輪廓、身形腰身、手指、走路、轉身回眸、喝酒、拔劍、戰鬥、觸碰、坐臥、壓抑、憤怒、恐懼、夜色燭火、章尾轉折等小分類的句子級模板。

載入寫作 AGENT 時，這五個欄位會一起寫進 `Technique Library`，讓寫作模型不只知道「要寫眼睛」，而是知道「眼睛要從落點開始、停半息、只給一個光濕細節」。

---

## 8. 存讀檔 Save / Load

- `Save Project`：保存 World／Story Background、Characters、Lorebook、Full Story、Story Memory、Style DNA、Few-Shot Samples、Chronicle、Technique Library，到 `book_output/story_studio_save_<timestamp>.json`。
- `Load Project JSON`：讀取先前保存的 JSON，還原主要欄位。

---

## 9. 技法回灌與檢閱 Skill / Technique Review

兩組功能：把 `full_report.md` 蒸餾回寫作 AGENT；以及檢閱 Skill 與 Technique 輸出。

### full_report.md → Compact Technique Library → Writing AGENT

- 上傳 `full_report.md`、貼路徑（如 `C:\path\to\full_report.md`）或貼內容。
- `/goal`：蒸餾目標（預設轉成可複製敘事技巧、場景調度規則、節奏轉折規則、人物推進方法、文風規則、Director Instruction）。
- `Dry Run / Local Extract`：本地蒸餾，不呼叫 API。
- `Distill full_report To Compact Library`：蒸餾，輸出到 `book_output/report_technique_libraries/<timestamp>/`。
- `Load Distilled Library To Writing AGENT` / `Load Latest Saved Library To Writing AGENT`：載入寫作 AGENT 欄位。
- `Load Mode`：一般建議 `Replace Technique Library + Append Memory/Director`。

### Review Skill / Technique（本地，不呼叫 API）

`Review Target` 可選 `Skill + Latest Technique`、`Skill Only`、`Latest Integrated Technique Book Library`、`Latest Technique Finder Library`、`Latest Focused Technique Sheet`、`Latest Chapter Craft Report`、`Latest Plot Ideation Report`、`Custom Markdown / JSON Path`。

檢閱會抽出新的深度區段（Anatomy Breakdown、Sentence Rhythm、Word Palette、Sensory Layering、Weak vs Strong）做品質確認，保存到 `book_output/skill_technique_reviews/<timestamp>/`。

---

## 10. 說明書 Manual

在 App 內直接顯示這份說明書。若你在外部編輯了 `AI_Book_Writer_Studio_面板說明書.md`，按 **重新載入說明書 Reload Manual** 即可更新顯示。

---

## 11. 故事技能 Story Skill（智慧網路蒸餾 + 原則技能）

這個分頁把參考來源蒸餾成 `story_skill.json`，內容是「敘事方式、故事推進原則、劇情發想思路、描寫技法工具箱」。它只保留怎麼寫，不保留來源的人名、地名、事件、設定或可辨識原文。

### Step 0 ｜ 智慧網路技巧探索 Smart Web Skill Scout

- `目標小說類型 / 讀者效果`：描述你想逼近的類型、氛圍與讀者感受。
- `我的小說片段`：貼上目前片段，系統會找缺口，例如壓力源不足、感官太少、對話沒有潛台詞、段尾缺鉤子。
- `種子 URL`：可貼公開章節或參考頁，每行一個。
- `Source Modes`：可只讀種子 URL，也可做公開網頁搜尋與 Pixiv 公開搜尋連結探索。
- `Allowed Domains`：需要限制來源時填網域，例如 `pixiv.net example.com`。
- `Dry Run`：本地診斷，不呼叫 Grok；關掉後用 Analysis 模型正式蒸餾。

合規規則：只抓公開 HTTP(S) 頁面，先檢查 `robots.txt`，不登入、不繞付費牆、不解 CAPTCHA；保存短摘錄索引與抽象技巧，不保存整篇小說。輸出在 `book_output/web_skill_distillations/<timestamp>/`，並會把新的 `story_skill.json` 直接放入 Story Skill 狀態。

### Step 1 ｜ 蒸餾技能 Distill Skill

上傳 TXT、貼參考小說正文，或給章節目錄 URL，蒸餾成同樣的 `story_skill.json`。正式模式使用 Analysis/Grok；Dry Run 可檢查流程。

### Step 2 ｜ 編排劇情 + 產生提示詞

輸入全新的故事種子，系統用目前 skill 編排原創劇情與技法綁定提示詞，輸出到 `book_output/story_skills/orchestrations/<timestamp>/`。完成後會自動載入寫作區。

### Step 3 ｜ 載入寫作區

`①b 直接載入技法到寫作區` 會只載入技法與原則，不編排新劇情；適合你自己主導情節。`③ 載入到寫作區` 則載入 Step 2 的提示詞、技法與推進原則。

---

## 建議工作流（端到端）

### A. 多本小說彙整成可查詢的深度技法書庫

1. 到 `2. 參考書架`，一次上傳多本小說 TXT、`full_report.md`、Technique Finder JSON，或貼資料夾路徑。
2. 按 `Add To Reference Shelf`；不想納入的來源可選取後 `Remove Selected References`。
3. 到 `7. 深度技法書庫`，按 `Build Technique Book From Reference Shelf`。
4. 用 `Search Query` 搜尋 `容顏`、`眼睛`、`手`、`喝酒`、`拔劍` 等，或直接選分類。
5. 按 `Load Search Results To Writing AGENT`。
6. 回到 `3. 寫作`，寫作 AGENT 會依載入的深度技法卡輔助生成。

### B. 從一本小說萃取技法並回灌寫作

1. 到 `5. 章節技法分析`，上傳小說 TXT，`Chapter Limit` 設 `0`，正式執行 `Analyze Chapter Craft`。
2. 到 `9. 技法回灌與檢閱`，在 `full_report Path` 貼入剛產出的 `full_report.md`。
3. 按 `Distill full_report To Compact Library` → `Load Distilled Library To Writing AGENT`。
4. 回到 `3. 寫作` 開始續寫。

### C. 針對特定場景找深度手法

1. 到 `7. 深度技法書庫` → `Focused Technique Sheet`。
2. 填 `Specific Scene / Action / Situation / Desired Effect`，可上傳參考文本。
3. 按 `Aggregate Techniques`，得到含「動作解剖、句法節奏、用詞調色盤、弱寫→強寫」的深度手法表。
4. 把產出的 Director Instruction 貼到 `3. 寫作` 的 `Director Instruction`。

### D. 做劇情編排

1. 到 `6. 劇情編排`，填 `Story Seed / Premise`，可加入參考文本、Style DNA、Story Chronicle。
2. 按 `Generate Plot Ideation`，把章節表或 Director Instruction 用於 `3. 寫作`。

### E. 目標類型／片段 → 自動找缺口 → 蒸餾成 skill

1. 到 `11. 故事技能` → `Step 0 智慧網路技巧探索`。
2. 填目標類型或貼上自己的片段；可加公開 URL 或限制 Allowed Domains。
3. 先用 `Dry Run` 看缺口報告；要更精細再關掉 Dry Run 用 Analysis 模型。
4. 按 `①b 直接載入技法到寫作區`，或繼續 Step 2 編排新故事。

---

## 注意事項

- 整本 TXT 沒有目錄時：在 `5. 章節技法分析` 把 `Chapter Limit` 設 `0`，用 `Max Chars / Chapter / Auto Chunk` 當自動切塊大小；`7. 深度技法書庫` 的單本蒸餾也用相同的無標題切塊。
- `full_report.md` 不要直接貼進寫作 prompt，請先在 `9. 檢閱` 蒸餾成 `Technique Library`。
- 多本小說要彙整成長期可查的技法書庫時，優先用 `2. 參考書架` ＋ `7. 深度技法書庫`，不要逐項手填。
- Pixiv 或其他平台若 `robots.txt`、登入、付費牆或反自動化限制不允許讀取，`11. 故事技能` 的網路探索會略過，不會嘗試繞過限制。
- `Technique Library` 是低優先級參考；最新的 `Director Instruction` 和當前故事上下文優先級更高。
- `Dry Run` 不花 API；正式 Grok 分析才會呼叫 API，且深度欄位更貼合你的實際參考文本。
- 主寫作 Studio 與獨立改寫工具是不同服務；不要把主寫作面板的 token 設定視為改寫工具設定。
