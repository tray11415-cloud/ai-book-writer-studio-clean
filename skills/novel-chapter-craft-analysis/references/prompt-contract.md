# Prompt Contract

The chapter-craft analyzer uses `/goal` as the task spine and asks every role to answer in Traditional Chinese with specific, observable evidence.

Source loading contract:

- Uploaded TXT, pasted fiction text, and static chapter-directory URLs are supported.
- If TXT or pasted text has recognizable chapter headings, preserve those chapters.
- If TXT or pasted text has no table of contents or chapter headings, auto-split the whole book into chunks using `Max Chars / Chapter / Auto Chunk`; `Chapter Limit = 0` means analyze the full book.

Specialist output contract:

1. 本章在小說中的功能
2. 可觀察到的寫作技巧，至少 5 點，每點引用短語或描述原文現象
3. 技巧如何影響讀者
4. 可學習的寫法模板
5. 若要強化，本章最值得調整的 2 點

Synthesis output contract:

- 章節核心作用
- 本章最重要的 8 個寫作技巧
- 技巧拆解：鋪陳、衝突、人物、節奏、文風
- 可複製寫作公式
- 仿寫練習題 3 題
- 修稿建議 3 點

Plot ideation contract:

1. 先萃取參考文本的技法 DNA，不抄原文、不延續原作情節。
2. 將故事種子與專案上下文整合成可連載的劇情引擎。
3. 讓每章有章節功能、主要事件、人物變化、衝突升級與章尾鉤子。
4. 輸出可直接寫成場景的場面清單。
5. 輸出可貼進 Interactive Writing 的 Director Instruction。

Scene technique contract:

1. 支援兩種輸出：整本小說蒸餾成 Technique Finder library，或聚焦「特定場景 + 特定動作 + 特定情境」的一張手法表。
2. 產出可操作手法，不輸出完整正文。
3. 必須拆出場景質地、動作節拍、情境壓力、感官配置、視角距離、鏡頭調度、句式節奏、對白/沉默處理。
4. 產出 10 條可複製寫法公式。
5. 產出 5 條可貼進 Interactive Writing 的 Director Instruction。
6. 若有參考文本，只蒸餾技法，不抄原文、不延續原作情節。
7. Whole-novel distillation must produce Technique Finder cards with scene, action, situation, reader effect, technique summary, formulas, and Director Instruction.

Integrated Technique Book Library contract:

- This is the main workflow when the user wants a searchable technique book instead of manually filling one technique at a time.
- It must expose a persistent Reference Shelf so the user can add uploaded files, add local paths/folders, add pasted text, remove selected references, clear the shelf, and reload saved references at any time.
- The shelf must be saved as `book_output/technique_reference_shelf.json`.
- It must accept many source files at once: uploaded MD/TXT/JSON, one path per line, folder paths, or pasted text.
- It must support raw novels, `full_report.md`, Technique Finder Markdown, and Technique Finder JSON.
- It must organize output into big categories and subcategories, including examples such as `容顏描寫 / 眼睛描寫`, `容顏描寫 / 嘴巴描寫`, `身材描寫 / 手與指節描寫`, `動作描寫 / 喝酒飲茶描寫`, `動作描寫 / 拔劍出招描寫`, `情緒描寫 / 壓抑情緒描寫`, `場景氛圍 / 夜色燭火描寫`, and `節奏章法 / 章尾轉折`.
- Each card must include category, subcategory, title, aliases, trigger keywords, summary, formula steps, applicable scenes, Director Instruction, source labels, evidence count, evidence signals, deep breakdown, detail lenses, micro techniques, common mistakes, and practice prompts.
- Deep cards must discuss how a technique works, how to observe and write the small physical details, and how to avoid shallow adjective-only output. Body-part and action cards must be concrete enough for cases such as eyes, mouth, hands/fingers, body posture, drinking, sword draw, walking, emotional restraint, and chapter hooks.
- Search must work by keyword and category/subcategory.
- Load must write only matched cards into the writing AGENT `Technique Library`, with optional Story Memory and Director Instruction inserts.
- Dry Run must be local and no-credit; formal distillation uses Analysis / Grok.
- Outputs must be written under `book_output/integrated_technique_libraries/<timestamp>/`.

Skill / Technique review contract:

- Review must be local and read-only; do not call model APIs.
- Review must show the Skill files, generated technique/report artifact path, headings, Director Instructions when present, technique-related lines, and a bounded preview.
- Review must surface deep technique sections when present: `Deep Breakdown`, `Detail Lenses`, `Micro Techniques`, `Common Mistakes`, and `Practice Prompts`.
- For Integrated Technique Book JSON, Review must report entry count and how many entries contain deep fields.
- Review must redact preview lines that reference underage sexual content so those source-report lines are not reused as writing prompt material.
- Custom Markdown/JSON paths must be supported for downloaded reports such as `full_report (1).md`.
- Review outputs must be written under `book_output/skill_technique_reviews/<timestamp>/`.

full_report to writing AGENT contract:

- `full_report.md` must be distilled before it is loaded into the writing AGENT; never inject the whole report into the writing prompt.
- Distillation output must include `Technique Library`, `Story Memory Insert`, and `Director Instruction Insert`.
- `Technique Library` must be a real writing reference field consumed by `generate_continuation`.
- Load buttons must support loading the current distilled library or the latest saved distilled library into the writing AGENT fields.
- Formal distillation uses Analysis / Grok; Dry Run / Local Extract must remain available for no-credit local tests.
- Distillation outputs must be written under `book_output/report_technique_libraries/<timestamp>/`.

Model routing contract:

- Writing route: NALANG plans/polishes and Local LoRA drafts prose.
- Analysis route: Grok analyzes references, Style DNA, Chronicle, and Chapter Craft.
- Integrated Technique Book route: Grok formally distills many sources into category/subcategory technique cards; Dry Run uses local categorization.
- Technique Finder route: Grok distills whole novels into card libraries and aggregates scene/action/situation depiction methods.
- Plot Ideation route: Grok + NALANG + Local LoRA all contribute; report which role used which route.
- Frontend route config must be editable and persisted with Writing, Analysis, and LoRA sections.
