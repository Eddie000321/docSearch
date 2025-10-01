from services.chunking import Segment, build_chunks


def test_chunk_overlap_and_size():
    text = " ".join(["evidence" for _ in range(200)])
    segments = [Segment(text=text, label="Paragraph 1", start_offset=0)]
    chunk_size = 120
    overlap = 30
    chunks = build_chunks(
        doc_id="doc-1",
        file_type="txt",
        segments=segments,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size

    for previous, current in zip(chunks, chunks[1:]):
        overlap_chars = previous.offset + len(previous.text) - current.offset
        assert overlap_chars >= 0
        assert current.offset >= previous.offset
