"""Small, explicit registry for packaged Qwen task instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    instructions: str
    allowed_tools: tuple[str, ...]


class SkillRegistry:
    """Loads static instructions; Python, not Markdown, enforces permissions."""

    _TOOLS = {
        "responder": ("search_images", "search_memories", "web_search", "web_fetch"),
        "memory_extractor": (),
        "memory_curator": (),
    }

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or Path(__file__).with_name("skills")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._TOOLS)

    def get(self, name: str, instructions_override: str | None = None) -> Skill:
        if name not in self._TOOLS:
            raise ValueError(f"Unknown Mosaic skill: {name}.")
        path = self._root / name / "INSTRUCTIONS.md"
        try:
            packaged_instructions = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(f"Packaged instructions for {name} are unavailable.") from error
        instructions = packaged_instructions
        first_line = packaged_instructions.splitlines()[0] if packaged_instructions else ""
        version = first_line.removeprefix("# ").strip() if first_line.startswith("# ") else "v0"
        if instructions_override is not None:
            instructions = instructions_override.strip()
            version = f"{version}+local"
        if not instructions:
            raise RuntimeError(f"Packaged instructions for {name} are empty.")
        return Skill(name, version, instructions, self._TOOLS[name])
