"""Text chunking with overlap.

Ported from jarvis_memory.py chunking logic.
"""


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap_lines: int = 2,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Max characters per chunk.
        overlap_lines: Number of lines from end of previous chunk
                       to prepend to next chunk.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline

        if current_len + line_len > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))
            # Keep last N lines as overlap
            overlap = current_lines[-overlap_lines:] if overlap_lines > 0 else []
            current_lines = list(overlap) + [line]
            current_len = sum(len(l) + 1 for l in current_lines)
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
