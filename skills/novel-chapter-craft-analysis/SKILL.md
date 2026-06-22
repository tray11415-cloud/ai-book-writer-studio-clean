---
name: novel-chapter-craft-analysis
description: Chapter-craft, plot-ideation, integrated technique-book-library, scene-technique, full_report-to-agent, and review workflow for ai-book-writer-clean. Use when analyzing novels for writing craft; batch-distilling many MD/TXT/JSON novel or report files into a searchable big-category/subcategory technique book library; querying depiction skills such as face, eyes, body, hands, drinking, combat, emotion, atmosphere, or chapter hooks; loading matched techniques into the writing AGENT; reviewing Skill/Technique outputs; brainstorming plot or scene methods; or maintaining the related Gradio tabs and modules.
---

# Novel Chapter Craft Analysis

## Chapter Craft Workflow

Use `chapter_craft_skill.py` as the executable implementation. Keep the feature scoped to the Book Writer studio unless the user explicitly asks to modify the separate rewrite agent.

1. Accept one source: uploaded `.txt`, pasted fiction text, or a static HTML chapter directory URL.
2. Split chapters with Chinese headings such as `第一章`, `第十回`, `第3節`, English `Chapter 1`, or numbered headings. If a TXT has no table of contents or chapter headings, fall back to auto chunks using `Max Chars / Chapter / Auto Chunk`.
3. Apply `/goal`; default to chapter-by-chapter craft analysis covering structure, pacing, character motion, conflict design, prose style, reusable techniques, and revision targets.
4. Run the AI cluster:
   - `結構編劇`: chapter function, hook, cause/effect, beats, turn, close.
   - `節奏讀者`: suspense density, information release, fast/slow rhythm, page-turn pressure.
   - `人物弧線分析師`: desire, resistance, choice, relationship tension, dialogue as action.
   - `文風技術編輯`: POV, sentence rhythm, imagery, sensory detail, dialogue, staging, tone.
   - `總編 Agent`: synthesize the cluster notes into techniques, formulas, exercises, and revision advice.
5. Write Markdown reports under `book_output/chapter_craft_reports/<timestamp>/`.

## Plot Ideation Workflow

Use `plot_ideation_skill.py` for story arrangement and brainstorming. It reuses the chapter-craft source loading and technique-distillation pattern, then converts the craft DNA plus the project's existing settings into a usable plot plan.

1. Accept a story seed/premise plus optional reference source: uploaded `.txt`, pasted reference text, or static HTML chapter directory URL.
2. Pull project context from the Gradio studio where available: World / Story Background, Characters, Lorebook, Story Memory, Style DNA, and Story Chronicle.
3. Distill optional reference chapters into technique DNA without copying plot or prose.
4. Run the plot ideation cluster:
   - `故事引擎策劃`: core sell, protagonist desire, long-term question, serialization fuel.
   - `章節編排師`: chapter order, story beats, each chapter's function, turns, hooks.
   - `衝突設計師`: external obstacles, inner contradictions, relationship tension, secrets, cost.
   - `場景發想師`: concrete scenes, staging, emotional beats, sensory detail, dialogue sparks.
   - `總編型劇情編排 Agent`: synthesize into a chapter table, scene list, craft rules, and Director Instructions.
5. Write Markdown/JSON outputs under `book_output/plot_ideation/<timestamp>/`.

## Scene Technique Workflow

Use `scene_technique_skill.py` when the user wants methods for depicting a specific scene, action, and situation. This is not full prose generation; it produces a reusable technique sheet and Director Instructions.

### Integrated Technique Book Library

Use `technique_library_builder.py` when the user wants one integrated workflow instead of filling individual technique fields.

1. Manage references through a persistent Reference Shelf before building the library:
   - add uploaded sources,
   - add local file or folder paths,
   - add pasted source text with a label,
   - load the saved shelf,
   - remove selected references,
   - clear the shelf.
2. Save the shelf at `book_output/technique_reference_shelf.json` so the interface can add or remove novel references anytime.
3. Accept many sources at once:
   - multiple uploaded `.md`, `.txt`, or `.json` files,
   - one local path per line,
   - a folder path that should be scanned for MD/TXT/JSON,
   - or pasted source text.
4. Sources may be raw novels, `full_report.md`, Technique Finder Markdown, or Technique Finder JSON.
5. Build a searchable hierarchy with big categories and subcategories, such as:
   - `容顏描寫 / 眼睛描寫`,
   - `容顏描寫 / 嘴巴描寫`,
   - `身材描寫 / 手與指節描寫`,
   - `動作描寫 / 喝酒飲茶描寫`,
   - `動作描寫 / 拔劍出招描寫`,
   - `情緒描寫 / 壓抑情緒描寫`,
   - `場景氛圍 / 夜色燭火描寫`,
   - `節奏章法 / 章尾轉折`.
6. Each technique card must include category, subcategory, title, aliases, trigger keywords, summary, formula steps, applicable scenes, Director Instruction, source labels, evidence count, evidence signals, deep breakdown, detail lenses, micro techniques, common mistakes, and practice prompts.
7. Deep analysis must explain how to write body parts and actions in a practical way, such as eye focus, mouth movement, hand/finger pressure, drinking beats, sword draw timing, and chapter hooks.
8. Dry Run must use local keyword categorization for no-credit tests. Formal distillation uses Analysis / Grok and must output JSON entries.
9. Write Markdown/JSON outputs under `book_output/integrated_technique_libraries/<timestamp>/`.
10. Support search by query and category, then load only the matched technique cards into `Technique Library`, with optional Story Memory and Director Instruction inserts.

### Whole-Novel Distillation

Use `distill_novel_to_technique_finder(...)` when the user wants to drop a whole novel into the Skill and have it organized into Technique Finder.

1. Accept one novel source: uploaded TXT, pasted novel text, or static HTML chapter directory URL.
2. Split chapters using the shared chapter loader. If the novel TXT has no table of contents or chapter headings, auto-split it into chunks before sending each chunk to Grok.
3. For each selected chapter, ask Grok to extract Technique Finder cards.
4. Each card must contain:
   - title,
   - source chapter,
   - scene,
   - action,
   - situation,
   - reader effect,
   - technique summary,
   - scene texture,
   - action beats,
   - sensory focus,
   - POV/camera,
   - sentence rhythm,
   - reusable formulas,
   - Director Instruction.
5. Write a searchable Markdown library and JSON card library under `book_output/scene_techniques/library/<timestamp>/`.

### Focused Technique Sheet

1. Accept target constraints:
   - Specific scene: location or environment.
   - Specific action: movement, gesture, physical act, exchange, combat beat, ritual, etc.
   - Specific situation: emotional/social/narrative pressure around the action.
   - Desired reader effect: suspense, oppression, intimacy, solemnity, danger, grief, etc.
2. Accept optional reference source: uploaded `.txt`, pasted reference text, or static HTML chapter directory URL.
3. Use Analysis / Grok routing to aggregate:
   - scene texture,
   - action beats,
   - situation pressure,
   - sensory distribution,
   - POV distance and camera blocking,
   - sentence rhythm and paragraph density,
   - dialogue or silence handling,
   - reusable formulas,
   - Director Instructions.
4. Write Markdown/JSON outputs under `book_output/scene_techniques/<timestamp>/`.

## full_report To Writing AGENT Workflow

Use `report_technique_distiller.py` when the user wants to turn a `full_report.md` chapter-craft report into a compact technique library that the writing AGENT actually reads.

1. Accept one report source:
   - uploaded `full_report.md`,
   - pasted report text,
   - a local Markdown/JSON/TXT path,
   - or the latest `book_output/chapter_craft_reports/<timestamp>/full_report.md`.
2. Distill the report into:
   - `Technique Library`: compact craft rules that are injected into the writing prompt as a low-priority reference,
   - `Story Memory Insert`: a short reminder that the writing AGENT should borrow technique only,
   - `Director Instruction Insert`: scene-level instructions that can be appended to the current writing instruction.
3. Use Analysis / Grok for formal distillation; keep Dry Run / Local Extract available for free local tests.
4. Never load the whole `full_report.md` into the writing prompt. Keep the library compact, safe, and reusable.
5. Write Markdown/JSON outputs under `book_output/report_technique_libraries/<timestamp>/`.

## Skill / Technique Review Workflow

Use `skill_technique_review.py` when the user wants to inspect whether the Skill and technique outputs exist, what they contain, and which technique/report artifact is currently available for reuse.

1. Review project Skill files:
   - `skills/novel-chapter-craft-analysis/SKILL.md`,
   - `skills/novel-chapter-craft-analysis/references/prompt-contract.md`,
   - `skills/novel-chapter-craft-analysis/agents/openai.yaml`.
2. Review generated technique/report artifacts:
   - latest Integrated Technique Book Library,
   - latest Technique Finder library,
   - latest focused scene technique sheet,
   - latest Chapter Craft full report,
   - latest Plot Ideation report,
   - or a custom Markdown/JSON path supplied by the user.
3. Extract headings, Director Instructions, deep technique blocks (`Deep Breakdown`, `Detail Lenses`, `Micro Techniques`, `Common Mistakes`, `Practice Prompts`), technique lines, JSON card counts, integrated-entry counts, deep-field coverage, and a bounded preview. Redact local preview lines that reference underage sexual content so they are not accidentally reused as writing prompt material.
4. Write a review Markdown report under `book_output/skill_technique_reviews/<timestamp>/`.

## Integration Rules

- Keep model routes separated:
  - Writing uses NALANG plus Local LoRA through the Writing Provider / Writing Pipeline controls.
  - Analysis uses Grok through the `Analysis / Grok Routing` controls.
  - Plot Ideation uses all three: Grok for craft DNA/strategy, NALANG for structure/final synthesis, and Local LoRA for scene-level writing instincts.
- Allow model routing to be changed from the frontend. Keep `Save Model Config` / `Load Model Config` wired to `book_output/model_config.json`, and include Writing, Analysis, and LoRA fields in that config.
- For xAI/Grok, use `https://api.x.ai/v1` as Base URL and the user-selected Grok model; never hard-code API keys.
- Keep the Gradio integration in `app_gradio.py` narrow: add or maintain only the `Chapter Craft Analysis`, `Plot Ideation`, and `Technique Finder` tabs and their event bindings.
- Keep the `Integrated Technique Book Library` workflow wired to `technique_library_builder.py`. It must manage a persistent Reference Shelf, accept many source files, add/remove references anytime, create deep category/subcategory technique cards, support search, and load matched cards into the writing AGENT.
- Keep the `Technique Finder` tab wired to `scene_technique_skill.py` and Analysis / Grok routing. It must include both whole-novel distillation and focused scene/action/situation sheets.
- Keep the `full_report.md -> Compact Technique Library -> Writing AGENT` interface wired to `report_technique_distiller.py`. It must output a compact Technique Library and include buttons to load it into `Technique Library`, `Story Memory`, and `Director Instruction`.
- Keep the `Skill / Technique Review` tab wired to `skill_technique_review.py`. It must be read-only against source artifacts and must not spend API credits.
- Do not change `rewrite_len_slider`, the rewrite app folder, or the separate rewrite surface unless explicitly requested.
- Keep URL crawling limited to normal public static HTML pages; do not bypass login, paywalls, or site protections.

## Verification

After changes, run:

```powershell
python -m py_compile app_gradio.py chapter_craft_skill.py plot_ideation_skill.py scene_technique_skill.py technique_library_builder.py report_technique_distiller.py skill_technique_review.py
python -c "from chapter_craft_skill import analyze_chapter_craft; print(analyze_chapter_craft(None, '雨聲落下。\\n\\n' * 3000, '', '', 0, 4000, True, '', 'http://127.0.0.1:8000/v1', 'test-model')[0])"
python -c "from plot_ideation_skill import generate_plot_ideation; print(generate_plot_ideation(None, '', '', '一名被流放的劍修回到雪都查明師門覆滅真相', '', '仙俠懸疑', '章節連載', 6, '繁體中文', 0, 4000, True, '', [], [], '', '', '', '', 'https://api.x.ai/v1', 'grok-4.3', '', 'http://127.0.0.1:8000/v1', 'nalang-xl-0826-10k', 'http://127.0.0.1:8010/v1', 'qwen3-4b-novel-lora')[0])"
python -c "from scene_technique_skill import aggregate_scene_techniques; print(aggregate_scene_techniques(None, '', '', '王城正殿', '侍從接近昏迷者', '祕密刺殺前的壓迫感', '危險、安靜、不可明說', '', '繁體中文', 0, 4000, True, '', 'https://api.x.ai/v1', 'grok-4.3')[0])"
python -c "from scene_technique_skill import distill_novel_to_technique_finder; print(distill_novel_to_technique_finder(None, '第一章 雨夜\n\n雨落在城門。', '', '', '繁體中文', 1, 2, 4000, True, '', 'https://api.x.ai/v1', 'grok-4.3')[0])"
python -c "from technique_library_builder import build_integrated_technique_book_library; print(build_integrated_technique_book_library(None, '', '眼睛描寫：目光與視線承接壓抑情緒。\n喝酒描寫：舉杯、抿酒、放杯三拍製造對話壓力。', '', '繁體中文', 4000, 3, True, '', 'https://api.x.ai/v1', 'grok-4.3')[0])"
python -c "from technique_library_builder import clear_reference_shelf, add_references_to_shelf, build_integrated_technique_book_library_from_shelf; clear_reference_shelf(); s,p,state,c=add_references_to_shelf(None, '', '眼睛描寫：目光落點、抬眼、避開。\n喝酒描寫：舉杯、抿酒、放杯。', 'test-reference', '', 4000); print(build_integrated_technique_book_library_from_shelf(state, '', '繁體中文', 3, True, '', 'https://api.x.ai/v1', 'grok-4.3')[0])"
python -c "from report_technique_distiller import distill_full_report_to_agent_library; print(distill_full_report_to_agent_library(None, '', '## 技巧\n- 用動作外化情緒', '', '繁體中文', 4000, 2000, True, '', 'https://api.x.ai/v1', 'grok-4.3')[0])"
python -c "from skill_technique_review import review_skill_and_technique; print(review_skill_and_technique('Skill + Latest Technique', '', 8000)[0])"
```

Use Dry Run for workflow checks so no API credits are spent.
