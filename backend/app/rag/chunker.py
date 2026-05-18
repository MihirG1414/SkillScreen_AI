def chunk_text(text: str, chunk_size_words: int = 800, overlap_words: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size_words <= overlap_words:
        raise ValueError("chunk_size_words must be greater than overlap_words.")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks
