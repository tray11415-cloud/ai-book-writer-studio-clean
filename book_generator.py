"""Book generation using the legacy AutoGen multi-agent pipeline.

LEGACY / CLI ONLY: This module is part of the older AutoGen book pipeline that is
reachable only via ``main.py`` (run directly from the command line). It is NOT used
by the live applications. The maintained, user-facing pipelines are ``app_gradio.py``
(the Gradio Studio UI) and ``web_app.py`` (the Flask web app), which use their own
generation logic (repetition_guard, technique libraries, etc.). Keep this module
correct and importable, but prefer the live apps for actual book generation.
"""
import autogen
from typing import Dict, List, Optional
import os
import time
import re
import json

class BookGenerator:
    def __init__(self, agents: Dict[str, autogen.ConversableAgent], agent_config: Dict, outline: List[Dict]):
        """Initialize with outline to maintain chapter count context"""
        self.agents = agents
        self.agent_config = agent_config
        self.output_dir = "book_output"
        self.chapters_memory = []  # Store chapter summaries
        self.max_iterations = 3  # Limit editor-writer iterations (used for group chat rounds)
        self.outline = outline  # Store the outline
        os.makedirs(self.output_dir, exist_ok=True)
        self._memory_file = os.path.join(self.output_dir, "chapters_memory.json")
        self._load_chapters_memory()

    def _load_chapters_memory(self) -> None:
        """Restore chapter summaries from disk so continuity survives a restart."""
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.chapters_memory = data
        except (FileNotFoundError, ValueError, OSError) as e:
            # Missing/corrupt memory file is non-fatal: start with an empty list.
            print(f"No prior chapter memory loaded ({e}); starting fresh.")

    def _save_chapters_memory(self) -> None:
        """Persist chapter summaries so a crash mid-generation does not lose context."""
        try:
            with open(self._memory_file, "w", encoding="utf-8") as f:
                json.dump(self.chapters_memory, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"Warning: could not persist chapter memory: {e}")

    def _clean_chapter_content(self, content: str) -> str:
        """Clean up chapter content by removing artifacts and chapter numbers"""
        # Remove chapter number references
        content = re.sub(r'\*?\s*\(Chapter \d+.*?\)', '', content)
        content = re.sub(r'\*?\s*Chapter \d+.*?\n', '', content, count=1)
        
        # Clean up any remaining markdown artifacts
        content = content.replace('*', '')
        content = content.strip()
        
        return content
    

    def initiate_group_chat(self) -> autogen.GroupChat:
        """Create a new group chat for the agents with improved speaking order"""
        outline_context = "\n".join([
            f"\nChapter {ch['chapter_number']}: {ch['title']}\n{ch['prompt']}"
            for ch in sorted(self.outline, key=lambda x: x['chapter_number'])
        ])

        messages = [{
            "role": "system",
            "content": f"Complete Book Outline:\n{outline_context}"
        }]

        writer_final = autogen.AssistantAgent(
            name="writer_final",
            system_message=self.agents["writer"].system_message,
            llm_config=self.agent_config
        )
        
        return autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["memory_keeper"],
                self.agents["writer"],
                self.agents["editor"],
                writer_final
            ],
            messages=messages,
            # One round per agent plus a buffer, derived from max_iterations so the
            # configured iteration budget actually has an effect.
            max_round=self.max_iterations + 2,
            speaker_selection_method="round_robin"
        )

    def _get_sender(self, msg: Dict) -> str:
        """Helper to get sender from message regardless of format"""
        return msg.get("sender") or msg.get("name", "")

    def _is_chapter_complete(self, messages: List[Dict]) -> bool:
        """Check (without side effects) whether the conversation produced a complete chapter.

        This only inspects state; it does NOT save anything. Saving is handled
        exclusively by ``_process_chapter_results``/``_save_chapter`` so there is a
        single, consistent save path that always receives the messages list.

        The tags checked here intentionally match what the agents in ``agents.py``
        actually emit: the memory keeper emits ``MEMORY UPDATE:``, the writer emits
        ``SCENE:`` / ``SCENE FINAL:``, and the editor emits ``FEEDBACK:``.
        """
        print("Verifying chapter completion...")
        sequence_complete = {
            'memory_update': False,
            'scene': False,
            'feedback': False,
            'scene_final': False,
        }
        chapter_content = None

        # Analyze full conversation
        for msg in messages:
            content = msg.get("content", "")

            if "MEMORY UPDATE:" in content:
                sequence_complete['memory_update'] = True
            # Check the most specific tag first to avoid the "SCENE" substring of
            # "SCENE FINAL" setting both flags from a single final message.
            if "SCENE FINAL:" in content:
                sequence_complete['scene_final'] = True
                chapter_content = content.split("SCENE FINAL:")[1].strip()
            elif "SCENE:" in content:
                sequence_complete['scene'] = True
            if "FEEDBACK:" in content:
                sequence_complete['feedback'] = True

        # A chapter is complete when the writer produced a final scene with content.
        # The other flags are informative but the final scene is the hard requirement.
        return bool(sequence_complete['scene_final'] and chapter_content)

    # Backwards-compatible alias for the previous method name.
    _verify_chapter_complete = _is_chapter_complete
    
    def _prepare_chapter_context(self, chapter_number: int, prompt: str) -> str:
        """Prepare context for chapter generation"""
        if chapter_number == 1:
            return f"Initial Chapter\nRequirements:\n{prompt}"
            
        context_parts = [
            "Previous Chapter Summaries:",
            *[f"Chapter {i+1}: {summary}" for i, summary in enumerate(self.chapters_memory)],
            "\nCurrent Chapter Requirements:",
            prompt
        ]
        return "\n".join(context_parts)

    def generate_chapter(self, chapter_number: int, prompt: str) -> None:
        """Generate a single chapter with completion verification"""
        print(f"\nGenerating Chapter {chapter_number}...")

        # Bounds check: the outline may have fewer chapters than requested.
        if chapter_number - 1 >= len(self.outline) or chapter_number < 1:
            raise ValueError(
                f"Chapter {chapter_number} not found in outline "
                f"(outline has {len(self.outline)} chapters)"
            )

        try:
            # Create group chat with reduced rounds
            groupchat = self.initiate_group_chat()
            manager = autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config=self.agent_config
            )

            # Prepare context
            context = self._prepare_chapter_context(chapter_number, prompt)
            chapter_prompt = f"""
            IMPORTANT: Wait for confirmation before proceeding.
            IMPORTANT: This is Chapter {chapter_number}. Do not proceed to next chapter until explicitly instructed.
            DO NOT END THE STORY HERE unless this is actually the final chapter ({self.outline[-1]['chapter_number']}).

            Current Task: Generate Chapter {chapter_number} content only.

            Chapter Outline:
            Title: {self.outline[chapter_number - 1]['title']}

            Chapter Requirements:
            {prompt}

            Previous Context for Reference:
            {context}

            Follow this exact sequence for Chapter {chapter_number} only:

            1. Memory Keeper: Context (MEMORY UPDATE)
            2. Writer: Draft (CHAPTER)
            3. Editor: Review (FEEDBACK)
            4. Writer Final: Revision (CHAPTER FINAL)

            Wait for each step to complete before proceeding."""

            # Start generation
            self.agents["user_proxy"].initiate_chat(
                manager,
                message=chapter_prompt
            )

            if not self._is_chapter_complete(groupchat.messages):
                raise ValueError(f"Chapter {chapter_number} generation incomplete")
        
            self._process_chapter_results(chapter_number, groupchat.messages)
            chapter_file = os.path.join(self.output_dir, f"chapter_{chapter_number:02d}.txt")
            if not os.path.exists(chapter_file):
                raise FileNotFoundError(f"Chapter {chapter_number} file not created")
        
            completion_msg = f"Chapter {chapter_number} is complete. Proceed with next chapter."
            self.agents["user_proxy"].send(completion_msg, manager)
            
        except Exception as e:
            print(f"Error in chapter {chapter_number}: {str(e)}")
            self._handle_chapter_generation_failure(chapter_number, prompt)

    def _extract_final_scene(self, messages: List[Dict]) -> Optional[str]:
        """Extract chapter content with improved content detection"""
        for msg in reversed(messages):
            content = msg.get("content", "")
            sender = self._get_sender(msg)

            # AutoGen groupchat messages carry the agent name in "name" (mapped by
            # _get_sender), but it may be empty in some contexts. The SCENE/SCENE FINAL
            # tags are the real signal, so accept writer agents OR any message that
            # carries those tags with no conflicting attribution.
            if sender in ["writer", "writer_final", ""] or "SCENE FINAL:" in content or "SCENE:" in content:
                # Handle complete scene content
                if "SCENE FINAL:" in content:
                    scene_text = content.split("SCENE FINAL:")[1].strip()
                    if scene_text:
                        return scene_text
                        
                # Fallback to scene content
                if "SCENE:" in content:
                    scene_text = content.split("SCENE:")[1].strip()
                    if scene_text:
                        return scene_text
                        
                # Handle raw content
                if len(content.strip()) > 100:  # Minimum content threshold
                    return content.strip()
                    
        return None

    def _handle_chapter_generation_failure(self, chapter_number: int, prompt: str) -> None:
        """Handle failed chapter generation with simplified retry"""
        print(f"Attempting simplified retry for Chapter {chapter_number}...")
        
        try:
            # Create a new group chat with just essential agents
            retry_groupchat = autogen.GroupChat(
                agents=[
                    self.agents["user_proxy"],
                    self.agents["story_planner"],
                    self.agents["writer"]
                ],
                messages=[],
                max_round=3
            )
            
            manager = autogen.GroupChatManager(
                groupchat=retry_groupchat,
                llm_config=self.agent_config
            )

            retry_prompt = f"""Emergency chapter generation for Chapter {chapter_number}.
            
{prompt}

Please generate this chapter in two steps:
1. Story Planner: Create a basic outline (tag: PLAN)
2. Writer: Write the complete chapter (tag: SCENE FINAL)

Keep it simple and direct."""

            self.agents["user_proxy"].initiate_chat(
                manager,
                message=retry_prompt
            )
            
            # Save the retry results
            self._process_chapter_results(chapter_number, retry_groupchat.messages)

        except Exception as e:
            print(f"Error in retry attempt for Chapter {chapter_number}: {str(e)}")
            # Do NOT swallow the failure: re-raise so the caller (generate_book)
            # can detect that the chapter was never produced and halt, rather than
            # silently skipping chapters and producing a truncated book.
            raise RuntimeError(
                f"Failed to generate chapter {chapter_number} after retry"
            ) from e

    def _process_chapter_results(self, chapter_number: int, messages: List[Dict]) -> None:
        """Process and save chapter results, updating memory"""
        try:
            # Extract the Memory Keeper's final summary
            memory_updates = []
            for msg in reversed(messages):
                sender = self._get_sender(msg)
                content = msg.get("content", "")
                
                if sender == "memory_keeper" and "MEMORY UPDATE:" in content:
                    update_start = content.find("MEMORY UPDATE:") + 14
                    memory_updates.append(content[update_start:].strip())
                    break
            
            # Add to memory even if no explicit update (use basic content summary)
            if memory_updates:
                self.chapters_memory.append(memory_updates[0])
            else:
                # Create basic memory from chapter content
                chapter_content = self._extract_final_scene(messages)
                if chapter_content:
                    basic_summary = f"Chapter {chapter_number} Summary: {chapter_content[:200]}..."
                    self.chapters_memory.append(basic_summary)

            # Persist accumulated memory so continuity survives interruptions.
            self._save_chapters_memory()

            # Extract and save the chapter content
            self._save_chapter(chapter_number, messages)
            
        except Exception as e:
            print(f"Error processing chapter results: {str(e)}")
            raise

    def _save_chapter(self, chapter_number: int, messages: List[Dict]) -> None:
        print(f"\nSaving Chapter {chapter_number}")
        # Guard against path traversal / malformed filenames: the chapter number is
        # interpolated into the output path, so require a positive integer.
        if not isinstance(chapter_number, int) or chapter_number < 1:
            raise ValueError(f"Invalid chapter number: {chapter_number!r}")
        try:
            chapter_content = self._extract_final_scene(messages)
            if not chapter_content:
                raise ValueError(f"No content found for Chapter {chapter_number}")
                
            chapter_content = self._clean_chapter_content(chapter_content)
            
            filename = os.path.join(self.output_dir, f"chapter_{chapter_number:02d}.txt")
            
            # Create backup if file exists
            if os.path.exists(filename):
                backup_filename = f"{filename}.backup"
                import shutil
                shutil.copy2(filename, backup_filename)
                
            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"Chapter {chapter_number}\n\n{chapter_content}")
                
            # Verify file
            with open(filename, "r", encoding='utf-8') as f:
                saved_content = f.read()
                if len(saved_content.strip()) == 0:
                    raise IOError(f"File {filename} is empty")
                    
            print(f"✓ Saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving chapter: {str(e)}")
            raise

    def generate_book(self, outline: List[Dict]) -> None:
        """Generate the book with strict chapter sequencing"""
        print("\nStarting Book Generation...")
        print(f"Total chapters: {len(outline)}")
        
        # Sort outline by chapter number
        sorted_outline = sorted(outline, key=lambda x: x["chapter_number"])
        
        for chapter in sorted_outline:
            chapter_number = chapter["chapter_number"]
            
            # Verify previous chapter exists and is valid
            if chapter_number > 1:
                prev_file = os.path.join(self.output_dir, f"chapter_{chapter_number-1:02d}.txt")
                if not os.path.exists(prev_file):
                    print(f"Previous chapter {chapter_number-1} not found. Stopping.")
                    break
                    
                # Verify previous chapter content
                with open(prev_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not self._verify_chapter_content(content, chapter_number-1):
                        print(f"Previous chapter {chapter_number-1} content invalid. Stopping.")
                        break
            
            # Generate current chapter
            print(f"\n{'='*20} Chapter {chapter_number} {'='*20}")
            self.generate_chapter(chapter_number, chapter["prompt"])
            
            # Verify current chapter
            chapter_file = os.path.join(self.output_dir, f"chapter_{chapter_number:02d}.txt")
            if not os.path.exists(chapter_file):
                print(f"Failed to generate chapter {chapter_number}")
                break
                
            with open(chapter_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not self._verify_chapter_content(content, chapter_number):
                    print(f"Chapter {chapter_number} content invalid")
                    break
                    
            print(f"✓ Chapter {chapter_number} complete")
            time.sleep(5)

    def _verify_chapter_content(self, content: str, chapter_number: int) -> bool:
        """Verify chapter content is valid"""
        if not content:
            return False
            
        # Check for chapter header
        if f"Chapter {chapter_number}" not in content:
            return False
            
        # Ensure content isn't just metadata
        lines = content.split('\n')
        content_lines = [line for line in lines if line.strip() and 'MEMORY UPDATE:' not in line]
        
        return len(content_lines) >= 3  # At least chapter header + 2 content lines