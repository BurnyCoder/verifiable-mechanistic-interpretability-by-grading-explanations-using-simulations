"""Plain-language glosses for the README: "E1" → "E1 (algorithm only)", "HS" → "HS (hidden-state similarity, …)".

Local explanation: `gloss(md)` adds a short meaning in parentheses after the first occurrence of each abbreviation in
every table row (one line) and every paragraph or list item. It leaves untouched: code spans, code fences, link texts
and targets, image lines, HTML comments, headings, abbreviations already followed by "(", and compound forms such as
"E3/E4", "E3>W-freq", "E1→E4" or "T0's", where an inserted gloss would break the text. Idempotent: glossed text is
left as it is.
Global context: `report.regenerate` passes every README block through `gloss`, so the generated tables read without
the legend; the paper and reports keep the compact labels for expert readers.
"""

from __future__ import annotations  # postponed annotations

import re  # tokenising markdown

GLOSS = {  # the shortest wording that still tells a non-expert what the label stands for
    "E-paper": "the paper's own explanation",
    "E0": "no explanation", "E1": "algorithm only", "E2": "+ key frequencies", "E3": "+ per-frequency details",
    "E4": "+ per-neuron table",
    "W-freq": "wrong frequencies", "W-alg": "wrong algorithm: Pizza", "W-role": "wrong component roles",
    "W-scramble": "shuffled neuron table",
    "T-paper": "the published checkpoint", "T-new": "target trained after the freeze", "T0": "main target network",
    "T1": "target from another seed", "T2": "target from another seed",
    "HS": "hidden-state similarity, 1 = internals match up to a linear map", "KL_out": "output divergence, 0 = same answers",
    "KL": "output divergence, 0 = same answers",
}
_TOKEN = re.compile(r"(?<![\w/\->→'`.–])(" + "|".join(re.escape(k) for k in sorted(GLOSS, key=len, reverse=True))
                    + r")(?![\w\-/>→'’`–])")  # en dash: ranges such as "E0–E4" are left as they are
_EXPLAINED = re.compile(r" ?\((?![\d.\-–])")  # "E1 (algorithm only)" is explained; "T2 (0.274)" is not
_LABELS = "|".join(re.escape(k) for k in sorted(GLOSS, key=len, reverse=True))
_COMPOUND = re.compile(rf"(?<![\w\-])({_LABELS})([>/→])(?=({_LABELS})(?![\w\-]))")  # E3>W-freq, E3/E4, E1→E4
_PROTECT = re.compile(r"`[^`]*`|\[[^\]]*\]\([^)]*\)|<[^>]+>|https?://\S+")  # code spans, links, html, urls


def _gloss_unit(text: str) -> str:
    """Gloss the first occurrence of each label in one unit (a table row or a paragraph), outside protected spans."""
    seen: set[str] = set()
    out, pos = [], 0
    spans = [(m.start(), m.end()) for m in _PROTECT.finditer(text)]

    def free(a: int, b: int) -> str:
        def repl(m):
            k = m.group(1)
            after, before = m.string[m.end(): m.end() + 50], m.string[max(0, m.start() - 45): m.start()]
            meaning = GLOSS[k].split(",")[0]  # e.g. "hidden-state similarity (HS)" is explained by the words before it
            if k in seen or _EXPLAINED.match(after) or meaning in after or meaning in before:  # already explained here
                seen.add(k)
                return k
            seen.add(k)
            return f"{k} ({GLOSS[k]})"
        spaced = _COMPOUND.sub(lambda m: m.group(1) + {">": " > ", "/": " and ", "→": " → "}[m.group(2)], text[a:b])
        return _TOKEN.sub(repl, spaced)

    for s, e in spans:
        out.append(free(pos, s))
        out.append(text[s:e])
        pos = e
    out.append(free(pos, len(text)))
    return "".join(out)


def gloss(md: str) -> str:
    """Gloss a markdown document or block: table rows one by one, prose paragraph by paragraph."""
    lines, out, para, fence = md.split("\n"), [], [], False

    def flush():
        if para:
            unit = "\n".join(para)
            keep = unit.lstrip().startswith("*Labels.*")  # the legend explains the labels itself
            out.extend((unit if keep else _gloss_unit(unit)).split("\n"))
            para.clear()

    for ln in lines:
        s = ln.lstrip()
        if s.startswith("```"):
            flush()
            fence = not fence
            out.append(ln)
        elif fence or s.startswith(("#", "![", "<!--", "|---")) or not s:
            flush()
            out.append(ln)
        elif s.startswith("|"):  # a table row is its own unit
            flush()
            out.append(_gloss_unit(ln))
        elif s.startswith(("- ", "* ", "> ")) and para:  # a new list item or quote line starts a new unit
            flush()
            para.append(ln)
        else:
            para.append(ln)
    flush()
    return "\n".join(out)
