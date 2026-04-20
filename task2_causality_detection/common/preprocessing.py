import re
import unicodedata


def clean_loose(text: str) -> str:
    """NFKC + whitespace collapse. Not offset-preserving — only for subtasks 1 & 3."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
