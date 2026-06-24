# AI Book Writer Studio

A Gradio-based studio for long-form Chinese fiction. It drives a writing model
(NALANG and/or a local LoRA) through an interactive, top-to-bottom pipeline:
core settings and story bible, a global reference shelf, interactive writing,
rewrite / style transfer, chapter-craft analysis, plot ideation, a searchable
"deep technique" library, save/load, and a technique review/feedback loop.

The studio talks to OpenAI-compatible endpoints through a small local
compatibility proxy (`compat_proxy.py`), so the same code path serves NALANG,
Grok (xAI), and a local LoRA server. Generation includes a cross-response
anti-repetition guard and long-form memory (see
[Cross-Response Repetition Guard](#cross-response-repetition-guard)).

> **Legacy:** the original AutoGen multi-agent book generator still ships in
> this repo (`main.py`, `agents.py`, `book_generator.py`,
> `outline_generator.py`, `config.py`) and remains usable. The sections below
> document the current Gradio studio; the legacy pipeline is summarized under
> [Legacy AutoGen pipeline](#legacy-autogen-pipeline).

## Features

- Single Gradio studio organized as a guided writing pipeline (11 tabs).
- Model routing across NALANG (writing), Grok/xAI (analysis), and a local LoRA.
- Interactive scene-by-scene writing and rewrite / style transfer.
- Per-chapter "chapter craft" analysis producing a deep `full_report.md`.
- A searchable **deep technique book** built from a global reference shelf.
- Cross-response repetition guard + extractive long-form memory for continuity.
- Save / load of sessions and model configuration to `book_output/`.

## Entry points

- **`app_gradio.py`** — the main Gradio studio. Run it directly to launch the
  full UI: `python app_gradio.py`. This is what `start_studio.bat` runs.
- **`studio_launcher.py`** — the Windows EXE/launcher front end. It picks an
  open local port, starts the studio via `start_studio.bat`, and opens the
  browser. This is the code behind `StartAIBookWriterStudio.exe`.
- **`web_app.py`** — a lightweight Flask chat UI for conversational
  scene-by-scene writing against the same model routing. Run with
  `python web_app.py` (or `start_web.bat`). Reply length is controlled by
  `BOOK_WRITER_CHAT_MAX_TOKENS`.

## Installation

1. Clone the repository.

2. Create a virtual environment (the project is developed against `.venv`):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your endpoints and keys:
```bash
cp .env.example .env   # On Windows: copy .env.example .env
```
Then edit `.env` and fill in your NALANG / Grok / LoRA base URLs, models, and
API keys. **Never commit `.env`** — it is gitignored, and only
`.env.example` (placeholders) should be tracked.

## Usage

Launch the Gradio studio:
```bash
python app_gradio.py
```
or, on Windows, double-click `StartAIBookWriterStudio.exe` (see
[Windows launcher](#windows-launcher)).

For the lightweight chat UI instead of the full studio:
```bash
python web_app.py
```

## Configuration

Endpoints, models, and API keys are read from `.env` (see `.env.example` for
the full list) and surfaced in the studio under `1. 設定 Core Settings`. Model
routing and the repetition guard can also be tuned via environment variables —
see [Model routing](#model-routing) and
[Cross-Response Repetition Guard](#cross-response-repetition-guard).

## Windows launcher

Double-click `StartAIBookWriterStudio.exe` to start the Gradio studio. The launcher chooses an open local port, runs `start_studio.bat`, and opens the browser automatically.

For a panel-by-panel guide, see [AI_Book_Writer_Studio_面板說明書.md](AI_Book_Writer_Studio_面板說明書.md).

To rebuild the launcher:

```bat
build_studio_launcher.bat
```

## Integrated workflow & tabs

The studio is organized as a single top-to-bottom pipeline. The tab order **is** the recommended order:

1. `1. 設定 Core Settings` — models + story bible.
2. `2. 參考書架 Reference Library` — a **global** shelf; add or remove novels/reports (MD/TXT/JSON) at any time. Every analysis pulls from here.
3. `3. 寫作 Interactive Writing` — generate with the loaded techniques.
4. `4. 改寫 Rewrite / Style Transfer`.
5. `5. 章節技法分析 Chapter Craft Analysis` — per-chapter deep report (`full_report.md`).
6. `6. 劇情編排 Plot Ideation`.
7. `7. 深度技法書庫 Deep Technique Book` — build a searchable hierarchy such as `容顏描寫 / 眼睛描寫`, `身材描寫 / 手與指節描寫`, `動作描寫 / 喝酒飲茶描寫` from the Reference Library, then load matched techniques into the writing AGENT.
8. `8. 存讀檔 Save / Load` — **named multi-slot saves**: bundle the story bible, memory, technique library, prompts, **and the distilled Story Skill** into a named slot; keep many, load/delete any from a dropdown (see [Named save slots](#named-save-slots)). Single-file JSON export/import remains under an accordion.
9. `9. 技法回灌與檢閱 Skill / Technique Review` — distill `full_report.md` into a compact `Technique Library`, and review the latest deep outputs locally.
10. `10. 說明書 Manual` — the full panel guide rendered in-app.
11. `11. 故事技能 Story Skill` — distill an input novel into a reusable, **plot-bound** writing skill, orchestrate an original technique-bound prompt from it, and load it straight into Interactive Writing (see [Story Skill](#story-skill-distill--orchestrate--write)).

## Model routing

- Writing: NALANG + Local LoRA.
- Analysis: Grok.
- Plot Ideation: Grok + NALANG + Local LoRA together.
- Deep Technique Book: distills the global Reference Library into a searchable hierarchy of **deep** technique cards.

## Cross-Response Repetition Guard

Novel models tend to recycle the same descriptive phrases, metaphors, and scene
套路 every time they are called again. The OpenAI `frequency_penalty` /
`presence_penalty` parameters only act *within a single completion*, so they do
nothing about the same clichés reappearing across separate continuations — which
is what keeps the **横向重复比例 (cross-response repetition ratio)** high.

`repetition_guard.py` is a pure-Python module (no extra API calls) that the
studio uses to keep long stories fresh and continuous:

- **Overused-phrase mining** (`extract_overused_phrases`) scans the whole story
  so far and surfaces the wordings that recur most, so the model can be handed
  an explicit "stop reusing these" avoid-list.
- **Overlap measurement** (`repetition_ratio`) scores how much a fresh
  continuation overlaps the prior story (0..1). When the ratio exceeds the
  threshold, the studio regenerates with a stronger ban (`repeated_spans` +
  `build_avoid_directive`), up to a configurable retry budget.
- **Long-form memory** (`build_longform_memory`) builds an extractive digest of
  earlier content that has scrolled out of the recent-context window, so long
  stories keep continuity without re-reading (and re-echoing) the full prose.

Phrase detection uses character n-grams for CJK text and word n-grams for
mostly-Latin text, so it handles Chinese (no spaces) and English alike.
Character names and proper nouns are explicitly exempt from the avoid-list —
the guard targets *phrasing*, not *people or objects*.

### Environment variables

All are optional; defaults are shown. Set `BOOK_WRITER_REPETITION_GUARD=0`
(or `false`/`off`/`no`) to disable the guard entirely.

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOOK_WRITER_REPETITION_GUARD` | `1` | Master on/off switch for the guard. |
| `BOOK_WRITER_REPETITION_THRESHOLD` | `0.30` | Overlap ratio above which a continuation is regenerated. |
| `BOOK_WRITER_REPETITION_MAX_RETRIES` | `1` | Max regeneration retries when overlap is too high. |
| `BOOK_WRITER_REPETITION_NGRAM` | `8` | CJK char n-gram size used to measure overlap. |
| `BOOK_WRITER_REPETITION_PHRASE_NGRAM` | `10` | CJK char n-gram size used to mine recurring 套路 phrases. |
| `BOOK_WRITER_REPETITION_MIN_COUNT` | `3` | Min recurrences before a phrase counts as overused. |
| `BOOK_WRITER_REPETITION_TOP_K` | `14` | How many overused phrases to feed back as an avoid-list. |

Some baseline overlap is normal (names, function words), so judge a threshold
change against a baseline run rather than against the absolute number. Lowering
the threshold or raising the retry count makes generations stricter but slower.

## Named save slots

Tab `8. 存讀檔 Save / Load` (module `project_saves.py`) keeps **many named saves**
instead of one file. A slot bundles the full working state — `background`,
characters, lorebook, full story, memory, Style DNA / samples, chronicle,
technique library, System Prompt Override, director note, story instruction —
**and the currently distilled Story Skill JSON**. Saves live under
`book_output/saves/<slug>.json`.

- **Save To Slot** — name it (optional note), and the current state + skill are
  written to that slot (re-saving the same name updates it).
- **Saved Slots dropdown + info table** — every slot with its time, story length,
  whether it carries a skill (and the skill's source / technique & beat counts),
  and your note.
- **Load Slot** — restores all fields *and* the skill, so you can jump to
  `11. 故事技能` and orchestrate / direct-load from the restored skill.
- **Delete Slot** — removes a save.

The older single-file JSON export/import is still available under the
Export / Import accordion.

## Story Skill (distill → orchestrate → write)

Tab `11. 故事技能 Story Skill` (module `story_skill_studio.py`) turns an input
novel into a reusable **writing skill** that *deeply binds plot arrangement to
description technique*, then writes an original story with it. The skill carries
the source's **craft, never its content** — no source characters, places,
objects, or concrete events — so the output is a fresh, original story, not a
continuation of or a remix of the source's 人事物.

Three stages, each with a Dry-Run mode for offline checking:

1. **Distill** (`distill_story_skill`) — feed a reference novel (TXT / pasted
   text / chapter-directory URL). The analysis model (Grok) returns a structured
   skill JSON:
   - `narrative_method` — POV, tense, narration distance, voice, dialogue style.
   - `description_techniques[]` — technique cards (when-to-use, how, sentence
     rhythm, sensory layering, word palette, weak-vs-strong).
   - `plot_arrangement` — engine, pacing curve, escalation, hook, arc shape
     (abstract patterns only).
   - `beat_template[]` — the **binding**: each beat names its `function`,
     `pacing`, `emotional_target`, and the `bound_technique_ids` (which
     description techniques to apply at that beat) plus how to apply them.
   The skill is saved to `book_output/story_skills/<timestamp>/story_skill.json`.

2. **Orchestrate** (`orchestrate_story_prompt`) — skill-driven Plot Ideation.
   Give it a **new** story seed; it invents an original plot (all-new people /
   places / events) following the skill's beat template, and composes a
   high-quality, beat-by-beat **technique-bound writing prompt** where every
   chapter says which techniques to apply. Saved under
   `book_output/story_skills/orchestrations/<timestamp>/`.

3. **Write** — the distill area (tab 11) and the writing area (tab 3) bind through
   a load action that pushes the skill into **System Prompt Override**,
   **Technique Library**, and **Story Memory**. Switch to
   `3. 寫作 Interactive Writing` and generate — the writing model now follows the
   source's narrative method and applies its techniques at the right beats, with
   entirely original content. Generation still runs through the
   [repetition guard](#cross-response-repetition-guard).

   There are two ways to bind, plus the manual button:
   - **Distill → write your own** (`①b 直接載入技法到寫作區`): right after Step 1,
     load *only* the craft (narrative method + technique cards + beat→technique
     mapping) — no invented plot. You drive the plot via `Story Instruction`, and
     the model applies the bound techniques at the matching beat type.
   - **Orchestrate → auto-load**: finishing Step 2 automatically loads the
     orchestrated, plot-bound prompt into the writing area (no manual click).
   - **Manual** (`③ 載入到寫作區` / `載入最近一次編排`): push the latest
     orchestration whenever you want.

   All three honor the **Load Mode** (Replace vs Append).

The abstraction barrier (extract patterns, drop entities; original content only)
is what separates this from `4. 改寫 Rewrite / Style Transfer`, which transforms
an existing passage in place.

### Continuation mode — continue your own novel with the skill

The Story Skill tab also has a **續寫模式 ｜ Continue an Existing Novel** section
(`read_continuation_source`). Attach the novel you actually want to **continue**
(TXT / pasted text / chapter-directory URL) and it reads the *continuation
context* — and, **unlike distillation, it keeps the source's real entities**,
because this is your own story to extend, not an abstract craft reference:

- The prose is loaded into **Full Story** (optionally capped to the most recent
  N characters — the part continuation writes from).
- With **extract brief** on, the analysis model pulls a structured brief —
  `background`, characters (→ the character table), plot-so-far, the current
  situation / where it left off, open threads, and tone — into **World /
  Background**, **Characters**, and an appended **Story Memory** block.
- A continue-from-here directive is placed in **Story Instruction**.

Combine the two halves of the tab: load a distilled skill's craft with
`①b 直接載入技法` (the *how*) **and** read a continuation source (the *what*),
then go to `3. 寫作` to continue your own novel in the distilled style.

## Deeper craft analysis

Every technique card now dissects **how a body part or an action is actually written**, not a few generic sentences. Each card carries five depth fields in addition to the base ones:

- `Anatomy Breakdown` — the body part / action split into observable micro-units in writing order (e.g. eyes = gaze-landing → eye-movement → dwell-time → one wet/light detail → the other person's reaction).
- `Sentence Rhythm` — clause length, pauses, punctuation cadence.
- `Word Palette` — concrete verbs/nouns/textures to use, and abstractions to avoid.
- `Sensory Layering` — which sense leads, in what order, how many details.
- `Weak vs Strong` — a short weak-vs-strong example sentence pair.

These appear in the Deep Technique Book (both the Grok and the offline Dry-Run paths), in the Focused Technique Sheet, and in the Chapter Craft synthesis report.

For whole-book TXT analysis without a table of contents, upload the TXT in `5. 章節技法分析 Chapter Craft Analysis`, set `Chapter Limit (0 = Full Book)` to `0`, and use `Max Chars / Chapter / Auto Chunk` as the automatic chunk size. The same no-heading TXT fallback is used by the single-novel distiller in `7. 深度技法書庫`.

To inspect what has actually been distilled, open `9. 技法回灌與檢閱 Skill / Technique Review`, choose `Skill + Latest Technique`, or paste a path such as `C:\Users\User\Downloads\full_report (1).md` under `Custom Markdown / JSON Path`.

To build the searchable technique book, first add novels/reports into `2. 參考書架 Reference Library` (remove selected references or clear the shelf at any time). Then open `7. 深度技法書庫`, click `Build Technique Book From Reference Shelf`, search for terms such as `眼睛`, `嘴巴`, `手`, `腰身`, `喝酒`, or choose a category like `動作描寫 / 喝酒飲茶描寫`, then click `Load Search Results To Writing AGENT`.

To feed chapter-craft analysis back into the writing AGENT, open `9. 技法回灌與檢閱 Skill / Technique Review`, use `full_report.md -> Compact Technique Library -> Writing AGENT`, distill the report, then click `Load Distilled Library To Writing AGENT`. The app writes the compact library to the `Technique Library` reference field and can append memory/director instructions for the next generation.

Model routing can be changed from the frontend in `1. 設定 Core Settings`:

- Edit Writing / NALANG Base URL, Model, and API key.
- Edit Analysis / Grok Base URL, Model, and API key.
- Edit LoRA Base URL and Model.
- Use `Save Model Config` and `Load Model Config` to persist these settings in `book_output/model_config.json`.

## Output Structure

Generated content is saved in the `book_output` directory:
```
book_output/
├── outline.txt
├── chapter_01.txt
├── chapter_02.txt
└── ...
```

## Requirements

- Python 3.8+
- AutoGen 0.2.0+
- Other dependencies listed in requirements.txt

## Development

To contribute to the project:

1. Fork the repository
2. Create a new branch for your feature
3. Install development dependencies:
```bash
pip install -r requirements.txt
```
4. Make your changes
5. Run tests:
```bash
pytest
```
6. Submit a pull request

## Error Handling

The system includes robust error handling:
- Validates chapter completeness
- Ensures proper formatting
- Maintains backup copies of generated content
- Implements retry logic for failed generations

## Limitations

- Requires significant computational resources
- Generation time increases with chapter count
- Quality depends on the underlying LLM model
- May require manual review for final polish

## Legacy AutoGen pipeline

The original multi-agent book generator (built on
[AutoGen](https://github.com/microsoft/autogen)) is still included and usable.
It composes specialized agents — Story Planner, World Builder, Memory Keeper,
Writer, Editor, and Outline Creator — to generate a structured book from a
prompt. Example:

```python
from config import get_config
from agents import BookAgents
from book_generator import BookGenerator
from outline_generator import OutlineGenerator

agent_config = get_config()

outline_agents = BookAgents(agent_config)
agents = outline_agents.create_agents()

outline_gen = OutlineGenerator(agents, agent_config)
outline = outline_gen.generate_outline(your_prompt, num_chapters=25)

book_agents = BookAgents(agent_config, outline)
agents_with_context = book_agents.create_agents()
book_gen = BookGenerator(agents_with_context, agent_config, outline)
book_gen.generate_book(outline)
```

The `autogen` dependency in `requirements.txt` exists for this pipeline.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Development tools (pytest, black, flake8, mypy) are listed under the optional
section of `requirements.txt`; install them only if you intend to lint/test.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the [AutoGen](https://github.com/microsoft/autogen) framework
- Inspired by collaborative writing systems
