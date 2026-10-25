"""Structure-aware, token-bounded chunking for the RAG index (docs/decisions.md, D30).

The character splitter (D14) cut chunks at a fixed size, so it could split a
formula from its definition, a table from its header, or a list mid-sentence,
and 20% of its chunks were longer than the embedding model reads (the rest
was silently ignored). This chunker instead:

- splits Markdown into sections by h1-h3 headings, as before (the section path
  is the citation's location), but keeps every line exactly as written;
- reads each section as blocks: paragraphs, list items, tables, code blocks
  and headings;
- packs whole blocks into chunks of at most `max_tokens`, counted with the
  embedding model's own tokenizer on the exact text it embeds, special tokens
  included;
- keeps a heading or a lead-in ending in ":" (also inside emphasis, as in
  "**Details:**") with the block that follows, and
  a block that starts lowercase (a sentence continued after a formula) with
  the block before;
- splits a block only if it alone is too long: a table between rows, with its
  header repeated; a code block between lines, re-fenced; prose between
  sentences, and only a single over-long sentence between words;
- merges a short trailing chunk into its neighbour in the same section, and a
  section with almost no text of its own (a document title, a parent heading)
  into the first chunk of its first subsection.

Each chunk records the source lines it covers (`start_line`, `end_line`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.documents import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SECTION_LEVELS = 3                     # h1-h3 start a section, like the character splitter's HEADERS
FENCE = re.compile(r"^\s*(```|~~~)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
LEAD_IN = re.compile(r":[*_]*$")      # "...:" or, with the colon inside emphasis, "**Label:**"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[*])")


@dataclass
class Piece:
    """Source text that is never split further, with the lines (1-based) it comes from."""
    text: str
    start: int
    end: int
    kind: str                      # heading, text, item, table, code
    glue_next: bool = False        # keep with the following piece (heading, lead-in ending in ":")
    glue_prev: bool = False        # keep with the previous piece (continues its sentence)
    gap: bool = False              # a blank line separated it from the previous piece in the source


@dataclass
class Section:
    path: str
    pieces: list[Piece] = field(default_factory=list)


def _sections(lines: list[str], markdown: bool) -> list[tuple[str, list[tuple[int, str]]]]:
    """(section path, [(line number, line)]) in document order; headings inside code blocks are ignored."""
    sections, path, current, in_code = [], {}, [], False
    def close():
        if current:
            label = " > ".join(path[level] for level in sorted(path))
            sections.append((label or "(no heading)", list(current)))
            current.clear()
    for number, line in enumerate(lines, start=1):
        if FENCE.match(line):
            in_code = not in_code
        match = HEADING.match(line) if markdown and not in_code else None
        if match and len(match.group(1)) <= SECTION_LEVELS:
            close()
            level = len(match.group(1))
            path = {k: v for k, v in path.items() if k < level}
            path[level] = match.group(2).strip()
        current.append((number, line))
    close()
    return sections


def _blocks(numbered: list[tuple[int, str]], markdown: bool) -> list[Piece]:
    """Group a section's lines into blocks; blank lines and horizontal rules only separate them."""
    blocks, i, gap = [], 0, False
    def add(kind, rows):
        nonlocal gap
        text = "\n".join(line for _, line in rows).strip("\n")
        if text.strip():
            stripped = text.rstrip()
            blocks.append(Piece(text, rows[0][0], rows[-1][0], kind, gap=gap,
                                glue_next=kind == "heading" or (kind in ("text", "item") and bool(LEAD_IN.search(stripped))),
                                glue_prev=kind == "text" and stripped[:1].islower()))
        gap = False
    while i < len(numbered):
        number, line = numbered[i]
        stripped = line.strip()
        if not stripped or (markdown and RULE.match(line)):
            gap, i = True, i + 1
        elif markdown and FENCE.match(line):
            j = i + 1
            while j < len(numbered) and not FENCE.match(numbered[j][1]):
                j += 1
            add("code", numbered[i:j + 1])
            i = j + 1
        elif markdown and stripped.startswith("|"):
            j = i
            while j < len(numbered) and numbered[j][1].strip().startswith("|"):
                j += 1
            add("table", numbered[i:j])
            i = j
        elif markdown and HEADING.match(line):
            add("heading", [numbered[i]])
            i += 1
        else:
            item = markdown and bool(LIST_ITEM.match(line))
            j = i + 1
            while j < len(numbered):
                nxt = numbered[j][1]
                if (not nxt.strip() or (markdown and (FENCE.match(nxt) or nxt.strip().startswith("|")
                                                     or HEADING.match(nxt) or LIST_ITEM.match(nxt) or RULE.match(nxt)))):
                    break
                j += 1
            add("item" if item else "text", numbered[i:j])
            i = j
    return blocks


def _join(pieces: list[Piece]) -> str:
    return "".join(("" if k == 0 else "\n\n" if p.gap else "\n") + p.text for k, p in enumerate(pieces))


def _line_of(text: str, offset: int, first: int) -> int:
    return first + text.count("\n", 0, offset)


def _split_words(piece: Piece, fits: Callable[[str], bool]) -> list[Piece]:
    """Last resort for a single over-long sentence or row: greedy runs of words."""
    parts, start = [], 0
    spans = [m.span() for m in re.finditer(r"\S+", piece.text)]
    while start < len(spans):
        end = start + 1
        while end < len(spans) and fits(piece.text[spans[start][0]:spans[end][1]]):
            end += 1
        a, b = spans[start][0], spans[end - 1][1]
        parts.append(Piece(piece.text[a:b], _line_of(piece.text, a, piece.start), _line_of(piece.text, b, piece.start),
                           piece.kind))
        start = end
    return parts


def _split_prose(piece: Piece, fits: Callable[[str], bool], first_fits: Callable[[str], bool]) -> list[Piece]:
    """Greedy runs of whole sentences; a sentence that is too long on its own is split between words."""
    bounds = [0] + [m.end() for m in SENTENCE_END.finditer(piece.text)] + [len(piece.text)]
    sentences = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if piece.text[bounds[k]:bounds[k + 1]].strip()]
    parts, k = [], 0
    while k < len(sentences):
        a, end = sentences[k][0], k + 1
        room = first_fits if not parts else fits
        while end < len(sentences) and room(piece.text[a:sentences[end][1]].strip()):
            end += 1
        b = sentences[end - 1][1]
        part = Piece(piece.text[a:b].strip(), _line_of(piece.text, a, piece.start), _line_of(piece.text, b, piece.start),
                     piece.kind)
        parts.extend([part] if fits(part.text) else _split_words(part, fits))
        k = end
    return parts


def _split_lines(piece: Piece, fits: Callable[[str], bool], first_fits: Callable[[str], bool], head: list[str],
                 tail: list[str], later_head: list[str] | None = None) -> list[Piece]:
    """Greedy runs of lines, each wrapped in `head` and `tail` (a table header, or code fences).

    Parts after the first start with `later_head` instead, if given (a split
    table repeats the heading it sits under, as well as its header row).
    """
    lines = piece.text.split("\n")
    body = lines[len(head):len(lines) - len(tail)]
    first_body = piece.start + len(head)
    later_head = head if later_head is None else later_head
    def wrap(part: int, a: int, b: int) -> str:
        return "\n".join((head if part == 0 else later_head) + body[a:b] + tail)
    spans, k = [], 0  # greedy runs of lines
    while k < len(body):
        end = k + 1
        room = first_fits if not spans else fits
        while end < len(body) and room(wrap(len(spans), k, end + 1)):
            end += 1
        spans.append((k, end))
        k = end
    # the same number of parts with the lines spread evenly, so the last part is not a short remainder
    even = [(len(body) * p // len(spans), len(body) * (p + 1) // len(spans)) for p in range(len(spans))]
    if all((first_fits if p == 0 else fits)(wrap(p, a, b)) for p, (a, b) in enumerate(even)):
        spans = even
    parts = []
    for p, (k, end) in enumerate(spans):
        text = wrap(p, k, end)
        part = Piece(text, piece.start if k == 0 else first_body + k,
                     piece.end if end == len(body) else first_body + end - 1, piece.kind)
        parts.extend([part] if fits(text) else _split_words(part, fits))
    return parts


def _split_block(piece: Piece, fits: Callable[[str], bool], lead: str = "", heading: str = "") -> list[Piece]:
    """A block that does not fit on its own, split at the safest boundaries for its kind.

    `lead` is the heading or lead-in that introduces the block: the first part
    is sized so that it still fits in one chunk together with it. `heading` is
    the section's heading line; the later parts of a split table repeat it, so
    a part with only rows of numbers still says what they are.
    """
    first_fits = (lambda text: fits(f"{lead}\n\n{text}")) if lead else fits
    if piece.kind == "table":
        lines = piece.text.split("\n")
        head = lines[:2] if len(lines) > 2 and TABLE_SEPARATOR.match(lines[1]) else []
        later = [heading, ""] + head if heading and fits(f"{heading}\n\n" + "\n".join(head + lines[len(head):len(head) + 1])) else head
        parts = _split_lines(piece, fits, first_fits, head, [], later)
    elif piece.kind == "code":
        lines = piece.text.split("\n")
        closed = len(lines) > 1 and FENCE.match(lines[-1])
        parts = _split_lines(piece, fits, first_fits, lines[:1], [lines[-1]] if closed else [])
    else:
        parts = _split_prose(piece, fits, first_fits)
    if parts:
        parts[0].gap, parts[0].glue_prev = piece.gap, piece.glue_prev
        parts[-1].glue_next = piece.glue_next
    return parts


def _units(pieces: list[Piece]) -> list[list[Piece]]:
    """Runs of pieces that should stay together (glued by headings, lead-ins and continued sentences)."""
    units = []
    for k, piece in enumerate(pieces):
        if units and (pieces[k - 1].glue_next or piece.glue_prev):
            units[-1].append(piece)
        else:
            units.append([piece])
    return units


def _pack(pieces: list[Piece], fits: Callable[[str], bool]) -> list[list[Piece]]:
    """Whole units into as few chunks as fit; a unit too long for one chunk is laid out piece by piece."""
    chunks: list[list[Piece]] = []
    for unit in _units(pieces):
        if chunks and fits(_join(chunks[-1] + unit)):
            chunks[-1].extend(unit)
        elif fits(_join(unit)):
            chunks.append(list(unit))
        else:
            for k, piece in enumerate(unit):
                if chunks and fits(_join(chunks[-1] + [piece])):
                    chunks[-1].append(piece)
                    continue
                run = []  # headings and lead-ins that introduce this piece move with it
                while k > 0 and chunks[-1] and chunks[-1][-1].glue_next and fits(_join([chunks[-1][-1], *run, piece])):
                    run.insert(0, chunks[-1].pop())
                if chunks and not chunks[-1]:
                    chunks.pop()
                chunks.append(run + [piece])
    return chunks


def _rebalance(chunks: list[list[Piece]], fits: Callable[[str], bool], small: Callable[[list[Piece]], bool]) -> None:
    """Merge a short last chunk into the one before it; if that does not fit, move whole units back into it."""
    if len(chunks) < 2 or not small(chunks[-1]):
        return
    if fits(_join(chunks[-2] + chunks[-1])):
        chunks[-2].extend(chunks.pop())
        return
    while small(chunks[-1]):
        units = _units(chunks[-2])
        if len(units) < 2:
            return
        last = units[-1]
        if not fits(_join(last + chunks[-1])) or small(chunks[-2][:-len(last)]):
            return
        del chunks[-2][-len(last):]
        chunks[-1][:0] = last


def split_structured(documents: list[Document], count_tokens: Callable[[str], int], max_tokens: int,
                     min_tokens: int = 0, include_section: bool = False) -> list[Document]:
    """Chunks of at most `max_tokens` (as counted by `count_tokens`, on the text the index embeds).

    `count_tokens` must count exactly what the embedding model reads, special
    tokens included. With `include_section`, the section path is counted too,
    because the index then embeds it before the chunk (D16).
    """
    if max_tokens <= 0 or not 0 <= min_tokens < max_tokens:
        raise ValueError("max_tokens must be positive and min_tokens in [0, max_tokens)")
    chunks = []
    for document in documents:
        source = document.metadata["source"]
        markdown = source.endswith(".md")
        sections = [Section(path, _blocks(rows, markdown))
                    for path, rows in _sections(document.page_content.splitlines(), markdown)]
        sections = [s for s in sections if s.pieces]
        for k, section in enumerate(sections):
            prefix = f"{section.path}\n" if include_section and section.path != "(no heading)" else ""
            def fits(text, prefix=prefix):
                return count_tokens(prefix + text) <= max_tokens
            def small(pieces, prefix=prefix):
                return not pieces or count_tokens(prefix + _join(pieces)) < min_tokens
            following = sections[k + 1] if k + 1 < len(sections) else None
            if (following and following.path.startswith(section.path + " > ") and small(section.pieces)
                    and fits(_join(section.pieces + following.pieces[:1]))):
                # a title or parent heading with almost no text: it introduces its first subsection
                section.pieces[-1].glue_next = True
                following.pieces[0].gap = True
                following.pieces[:0] = section.pieces
                continue
            heading = next((p.text for p in reversed(section.pieces) if p.kind == "heading"
                            and HEADING.match(p.text) and len(HEADING.match(p.text).group(1)) <= SECTION_LEVELS), "")
            pieces: list[Piece] = []
            for piece in section.pieces:
                lead = []  # the headings and lead-ins directly before this block
                while len(lead) < len(pieces) and pieces[-1 - len(lead)].glue_next:
                    lead.insert(0, pieces[-1 - len(lead)])
                if not lead and pieces and small(_units(pieces)[-1]):
                    lead = _units(pieces)[-1]  # a short introduction, such as a heading and one sentence
                lead_text = _join(lead) if lead and fits(_join(lead)) else ""
                if fits(piece.text) and (not lead_text or fits(f"{lead_text}\n\n{piece.text}")):
                    pieces.append(piece)
                else:  # too long, or too long to stay with what introduces it
                    pieces.extend(_split_block(piece, fits, lead_text, heading))
            packed = _pack(pieces, fits)
            _rebalance(packed, fits, small)
            for group in packed:
                chunks.append(Document(page_content=_join(group), metadata={
                    "source": source, "section": section.path, "source_sha256": document.metadata["sha256"],
                    "start_line": group[0].start, "end_line": group[-1].end,
                }))
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#{counters[source]}"
    return chunks
