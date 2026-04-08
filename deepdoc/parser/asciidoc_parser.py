# -*- coding: utf-8 -*-
#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import re


# Lines that open or close an AsciiDoc delimited block. The delimiter line
# must consist of 4+ repetitions of one of these characters (or exactly two
# dashes for the open block).
DELIMITED_BLOCK_CHARS = {
    "-": "listing",  # ---- (also used for [source])
    ".": "literal",  # ....
    "=": "example",  # ==== (block, not header — headers always have a space)
    "*": "sidebar",  # ****
    "_": "quote",  # ____
    "+": "passthrough",  # ++++
}

# Matches a delimited block fence: 4+ identical chars, optional trailing space.
_DELIM_BLOCK_RE = re.compile(r"^([-.=*_+])\1{3,}\s*$")
# Matches the open block fence (exactly two dashes).
_OPEN_BLOCK_RE = re.compile(r"^--\s*$")
# Matches an AsciiDoc section title (= ... ====== ...). Must have a space
# after the leading equals so we don't confuse it with an `====` example
# block fence.
_HEADER_RE = re.compile(r"^={1,6}\s+\S")
# AsciiDoc table fences: |===, !===, ,===, :===
_TABLE_FENCE_RE = re.compile(r"^[|!,:]===\s*$")
# Unordered list item: leading spaces, then * or - markers and a space.
_UL_ITEM_RE = re.compile(r"^\s*(?:\*+|-)\s+\S")
# Ordered list item: leading spaces, then . markers or `1.` markers.
_OL_ITEM_RE = re.compile(r"^\s*(?:\.+|\d+\.)\s+\S")
# Block-image macro line: image::path[alt]
_BLOCK_IMAGE_RE = re.compile(r"^image::\S+\[[^\]]*\]\s*$")


def _is_block_start(line: str) -> bool:
    """Return True if the line starts a recognized block element."""
    if not line.strip():
        return False
    if _HEADER_RE.match(line):
        return True
    if _DELIM_BLOCK_RE.match(line):
        return True
    if _OPEN_BLOCK_RE.match(line):
        return True
    if _TABLE_FENCE_RE.match(line):
        return True
    if _UL_ITEM_RE.match(line) or _OL_ITEM_RE.match(line):
        return True
    if _BLOCK_IMAGE_RE.match(line):
        return True
    if line.startswith("[") and line.rstrip().endswith("]"):
        # Block attribute line, e.g. [source,python]
        return True
    return False


class RAGFlowAsciidocParser:
    def __init__(self, chunk_token_num=128):
        self.chunk_token_num = int(chunk_token_num)

    def _table_to_html(self, raw_table: str) -> str:
        """Render a raw AsciiDoc |=== table to a minimal HTML table.

        Cells are split on `|`. The first non-empty data row becomes the
        header row, matching the markdown library's table-extension behavior.
        """
        lines = [ln for ln in raw_table.splitlines() if ln.strip()]
        # Drop the opening/closing fences.
        lines = [ln for ln in lines if not _TABLE_FENCE_RE.match(ln)]
        rows: list[list[str]] = []
        current: list[str] = []
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("|"):
                # New row begins. Flush the previous row if any cells were
                # accumulated, then split this line on `|`.
                if current:
                    rows.append(current)
                    current = []
                # Split and drop the empty leading cell from the leading `|`.
                cells = [c.strip() for c in stripped.split("|")[1:]]
                current.extend(cells)
            else:
                # Continuation of the previous cell — join with a space.
                if current:
                    current[-1] = (current[-1] + " " + stripped).strip()
        if current:
            rows.append(current)

        if not rows:
            return "<table></table>"

        def _row_html(cells: list[str], tag: str) -> str:
            return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

        header = _row_html(rows[0], "th")
        body = "".join(_row_html(r, "td") for r in rows[1:])
        return "<table>" + header + body + "</table>"

    def extract_tables_and_remainder(self, asciidoc_text, separate_tables=True):
        """Pull AsciiDoc |=== tables out of the source.

        Returns ``(remainder_text, tables)``. When ``separate_tables`` is
        True, the table is removed from the remainder; otherwise it is
        replaced with the rendered HTML table so downstream chunking can
        still see it inline.
        """
        tables: list[str] = []
        lines = asciidoc_text.splitlines(keepends=True)
        out: list[str] = []

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            if _TABLE_FENCE_RE.match(line.rstrip("\n")):
                # Find the matching closing fence.
                end = i + 1
                while end < n and not _TABLE_FENCE_RE.match(lines[end].rstrip("\n")):
                    end += 1
                # Include the closing fence if found, otherwise consume to EOF.
                last = end if end < n else n - 1
                raw_table = "".join(lines[i : last + 1])
                tables.append(raw_table)
                if separate_tables:
                    out.append("\n\n")
                else:
                    out.append(self._table_to_html(raw_table) + "\n\n")
                i = last + 1
                continue
            out.append(line)
            i += 1

        return "".join(out), tables


class AsciidocElementExtractor:
    def __init__(self, asciidoc_content: str):
        self.asciidoc_content = asciidoc_content
        self.lines = asciidoc_content.split("\n")

    def get_delimiters(self, delimiters: str) -> str:
        toks = re.findall(r"`([^`]+)`", delimiters)
        toks = sorted(set(toks), key=lambda x: -len(x))
        return "|".join(re.escape(t) for t in toks if t)

    def extract_elements(self, delimiter=None, include_meta=False):
        """Extract individual elements (headers, code blocks, lists, ...).

        Mirrors :meth:`MarkdownElementExtractor.extract_elements`.
        """
        sections: list = []

        dels = ""
        if delimiter:
            dels = self.get_delimiters(delimiter)
        if len(dels) > 0:
            text = "\n".join(self.lines)
            if include_meta:
                pattern = re.compile(dels)
                last_end = 0
                for m in pattern.finditer(text):
                    part = text[last_end : m.start()]
                    if part and part.strip():
                        sections.append(
                            {
                                "content": part.strip(),
                                "start_line": text.count("\n", 0, last_end),
                                "end_line": text.count("\n", 0, m.start()),
                            }
                        )
                    last_end = m.end()

                part = text[last_end:]
                if part and part.strip():
                    sections.append(
                        {
                            "content": part.strip(),
                            "start_line": text.count("\n", 0, last_end),
                            "end_line": text.count("\n", 0, len(text)),
                        }
                    )
            else:
                parts = re.split(dels, text)
                sections = [p.strip() for p in parts if p and p.strip()]
            return sections

        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # Skip line comments and comment blocks first.
            if line.startswith("////"):
                end = i + 1
                while end < len(self.lines) and not self.lines[end].startswith("////"):
                    end += 1
                i = end + 1
                continue
            if line.startswith("//"):
                i += 1
                continue

            if _HEADER_RE.match(line):
                element = self._extract_header(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif _DELIM_BLOCK_RE.match(line) or _OPEN_BLOCK_RE.match(line):
                element = self._extract_delimited_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.startswith("[") and line.rstrip().endswith("]") and i + 1 < len(self.lines) and (_DELIM_BLOCK_RE.match(self.lines[i + 1]) or _OPEN_BLOCK_RE.match(self.lines[i + 1])):
                # Block attribute line followed by a delimited fence — emit
                # both as a single delimited block element.
                inner = self._extract_delimited_block(i + 1)
                element = {
                    "type": inner["type"],
                    "content": line + "\n" + inner["content"],
                    "start_line": i,
                    "end_line": inner["end_line"],
                }
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif _UL_ITEM_RE.match(line) or _OL_ITEM_RE.match(line):
                element = self._extract_list_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            elif line.strip():
                element = self._extract_text_block(i)
                sections.append(element if include_meta else element["content"])
                i = element["end_line"] + 1
            else:
                i += 1

        if include_meta:
            sections = [section for section in sections if section["content"].strip()]
        else:
            sections = [section for section in sections if section.strip()]
        return sections

    def _extract_header(self, start_pos: int) -> dict:
        return {
            "type": "header",
            "content": self.lines[start_pos],
            "start_line": start_pos,
            "end_line": start_pos,
        }

    def _extract_delimited_block(self, start_pos: int) -> dict:
        """Consume a delimited block, matching its opening fence to a close."""
        opener = self.lines[start_pos].rstrip()
        content_lines = [self.lines[start_pos]]
        end_pos = start_pos
        for i in range(start_pos + 1, len(self.lines)):
            content_lines.append(self.lines[i])
            end_pos = i
            if self.lines[i].rstrip() == opener:
                break

        # Pick a friendly type label from the opener char.
        first_char = opener[:1]
        block_type = DELIMITED_BLOCK_CHARS.get(first_char, "delimited_block")
        if _OPEN_BLOCK_RE.match(opener):
            block_type = "open"
        return {
            "type": block_type,
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_list_block(self, start_pos: int) -> dict:
        end_pos = start_pos
        content_lines = []

        i = start_pos
        while i < len(self.lines):
            line = self.lines[i]
            if _UL_ITEM_RE.match(line) or _OL_ITEM_RE.match(line) or (i > start_pos and not line.strip()) or (i > start_pos and re.match(r"^\s+\S", line)):
                # Stop on a blank line followed by a non-list block start.
                if i > start_pos and not line.strip():
                    if (
                        i + 1 < len(self.lines)
                        and self.lines[i + 1].strip()
                        and not (_UL_ITEM_RE.match(self.lines[i + 1]) or _OL_ITEM_RE.match(self.lines[i + 1]) or re.match(r"^\s+\S", self.lines[i + 1]))
                    ):
                        break
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                break

        return {
            "type": "list_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }

    def _extract_text_block(self, start_pos: int) -> dict:
        """Extract paragraphs/inline content until the next block element."""
        end_pos = start_pos
        content_lines = [self.lines[start_pos]]

        i = start_pos + 1
        while i < len(self.lines):
            line = self.lines[i]
            if _is_block_start(line):
                break
            if not line.strip():
                if i + 1 < len(self.lines) and _is_block_start(self.lines[i + 1]):
                    break
                content_lines.append(line)
                end_pos = i
                i += 1
            else:
                content_lines.append(line)
                end_pos = i
                i += 1

        return {
            "type": "text_block",
            "content": "\n".join(content_lines),
            "start_line": start_pos,
            "end_line": end_pos,
        }
