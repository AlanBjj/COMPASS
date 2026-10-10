"""Per-column heat-map tints for the paper tables.

Every tinted column is colored on its own scale. A column is tinted when its cells use \\hc (higher is better), \\hcl
(lower is better), or \\hcls (a signed difference, lower is better). Within the column, the median is white, and the other values are shaded by rank, one step per distinct
value away from the median: blue on the better side and orange on the worse side, with the fourth step and beyond at the
darkest shade. The numbers carry the magnitude and the shade carries the order. When the rows are grouped by backbone
(\\grp{..}{Qwen3-8B} ...), every column is scaled within each backbone, because absolute levels differ between backbones.
scope="row" scales each row instead (tables whose columns are methods).

The best value of each block is set in bold and the second best underlined, ties sharing a mark (set_marks=False keeps
the generator's marks). Cells printed with \\reused (values copied from
the full model in the ablation table) are not tinted.

The output is a prefix \\heat{t} in front of each colored cell, t in [-1, 1] (+1 darkest blue, -1 darkest orange); the
macro is defined in the paper's macros.tex. Deterministic and idempotent, one table row per line.
    from tab_tint import tint_fragment;  text = tint_fragment(text)        # a fragment of table rows
    python3 scripts/tab_tint.py [--scope block|column|row] file.tex [...]  # a complete tabular, rewritten in place
"""
import re, sys, statistics
from pathlib import Path

SCOPE = "block"
LEVELS = (0.0, 0.32, 0.55, 0.78, 1.0)   # shade of step 0 (the median) .. step 4 and beyond, as a fraction of the darkest tint
TINT = re.compile(r"\\(hcls|hcl|hca|hc)(?=[\[{])(\[[^\]]*\])?\{\s*([-+0-9.]+)\s*\}\{\s*([-+0-9.]+)\s*\}")
OLD = re.compile(r"\\tscale\{[^{}]*\}\{[^{}]*\}|\\heat\{[^{}]*\}")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
SKIP = ("\\same", "\\na", "\\multicolumn", "\\cost", "\\tbd", "\\reused")
MODEL = re.compile(r"\\grp\{\d+\}\{[^}]*(Llama|LLaMA|Ministral|Mistral|Qwen|Gemma)", re.I)
SIGN = {"hc": 1.0, "hca": 1.0, "hcl": -1.0, "hcls": -1.0}   # hcls: a signed difference, lower is better


def _split_cells(line):
    """Split a table row at top-level & (not inside braces, not \\&)."""
    cells, depth, start, i = [], 0, 0, 0
    while i < len(line):
        ch = line[i]
        if ch == "\\":
            i += 2; continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0:
            cells.append(line[start:i]); start = i + 1
        i += 1
    cells.append(line[start:])
    return cells


def _value(cell):
    """The number shown in a cell: the first argument of a tint macro, else the first number outside superscripts."""
    m = TINT.search(cell)
    if m:
        return float(m.group(3))
    if any(k in cell for k in SKIP):
        return None
    text = re.sub(r"\$\^\{[^}]*\}\$|\^\{[^}]*\}", "", cell)          # drop marks such as $^{\dagger}$
    n = NUM.search(text)
    return float(n.group(0)) if n else None


FMTWRAP = re.compile(r"\\(?:textbf|underline)\{([^{}]*)\}")


def _set_fmt(cell, fmt):
    """Put fmt (\\textbf, \\underline or '') on the number of a cell, removing any mark it had."""
    m = TINT.search(cell)
    if m:
        rep = "\\" + m.group(1) + (f"[{fmt}]" if fmt else "") + "{" + m.group(3) + "}{" + m.group(4) + "}"
        return cell[:m.start()] + rep + cell[m.end():]
    c = FMTWRAP.sub(lambda mm: mm.group(1), cell)
    text = re.sub(r"\$\^\{[^}]*\}\$", lambda mm: "\0" * len(mm.group(0)), c)    # do not touch numbers in superscripts
    n = NUM.search(text)
    if not n or not fmt:
        return c
    return c[:n.start()] + f"{fmt}{{{n.group(0)}}}" + c[n.end():]


def tint_columns(text, scope=None, set_marks=True):
    scope = scope or SCOPE
    lines = text.split("\n")
    tab, block, rows, in_tab, in_body = 0, 0, [], False, False
    header, begin = {}, {}
    for n, line in enumerate(lines):
        if "\\begin{tabular" in line:
            tab += 1; block = 0; in_tab = True; in_body = False; begin[tab] = n
        if in_tab and not in_body and "\\midrule" in line:
            in_body = True
        if in_tab and not in_body and "&" in line and "\\multicolumn" not in line:
            header[tab] = n                                   # the last column-name row before \midrule
        if in_tab and in_body and scope == "block" and MODEL.search(line):
            block += 1
        if in_tab and in_body and "&" in line and "\\multicolumn" not in line.split("&")[0]:
            rows.append((n, tab, block))
        if "\\end{tabular" in line:
            in_tab = False
    # which columns are tinted, and in which direction
    direction, ncols = {}, {}
    for n, t, b in rows:
        cells = _split_cells(lines[n]); ncols[t] = max(ncols.get(t, 0), len(cells))
        for c, cell in enumerate(cells):
            for m in TINT.finditer(cell):
                direction.setdefault((t, c), []).append(SIGN[m.group(1)])
    direction = {k: (1.0 if sum(v) >= 0 else -1.0) for k, v in direction.items()}
    key = (lambda t, b, c, n: (t, "row", n)) if scope == "row" else (lambda t, b, c, n: (t, b, c))
    vals = {}
    for n, t, b in rows:
        for c, cell in enumerate(_split_cells(lines[n])):
            if c == 0 or (t, c) not in direction:
                continue
            v = _value(OLD.sub("", cell))
            if v is not None:
                vals.setdefault(key(t, b, c, n), []).append((v, direction[(t, c)]))
    stats, marks = {}, {}
    for k, vs in vals.items():
        gs = [round(v * sgn, 6) for v, sgn in vs]                              # "goodness": higher is better after the sign
        med = statistics.median(gs)
        stats[k] = (med, sorted({g for g in gs if g > med}), sorted({g for g in gs if g < med}, reverse=True))
        good = sorted(set(gs), reverse=True)
        marks[k] = (good[0], good[1] if len(good) > 1 else None) if len(good) > 1 else (None, None)
    for n, t, b in rows:
        cells = _split_cells(lines[n]); out = []
        for c, cell in enumerate(cells):
            clean = OLD.sub("", cell)
            if c == 0 or (t, c) not in direction or key(t, b, c, n) not in stats:
                out.append(clean); continue
            v = _value(clean)
            if v is None:
                out.append(clean); continue
            k = key(t, b, c, n); med, above, below = stats[k]; s = direction[(t, c)]
            g = round(v * s, 6); best, second = marks[k]
            if set_marks:
                clean = _set_fmt(clean, r"\textbf" if g == best else (r"\underline" if g == second else ""))
            if g > med:
                x = LEVELS[min(above.index(g) + 1, len(LEVELS) - 1)]
            elif g < med:
                x = -LEVELS[min(below.index(g) + 1, len(LEVELS) - 1)]
            else:
                x = 0.0
            lead = len(clean) - len(clean.lstrip(" "))
            out.append(clean[:lead] + rf"\heat{{{x:g}}}" + clean[lead:])
        lines[n] = "&".join(out)
    # direction arrows in the column-name row
    for t, n in header.items():
        cells = _split_cells(lines[n])
        for c in range(1, len(cells)):
            if (t, c) in direction and "arrow" not in cells[c]:
                arrow = r"\,$\uparrow$" if direction[(t, c)] > 0 else r"\,$\downarrow$"
                body = cells[c].rstrip(); tail = cells[c][len(body):]
                if body.endswith("\\\\"):
                    body, tail = body[:-2].rstrip(), " \\\\" + tail
                cells[c] = body + arrow + tail
        lines[n] = "&".join(cells)
    # a tinted last column: end the tabular with one \tabcolsep so the colour stops at the rules
    for t, n in begin.items():
        if (t, ncols.get(t, 0) - 1) in direction and "@{\\hspace{\\tabcolsep}}}" not in lines[n]:
            i = lines[n].index("\\begin{tabular}{") + len("\\begin{tabular}{")
            depth, j = 1, i
            while depth:
                depth += {"{": 1, "}": -1}.get(lines[n][j], 0); j += 1
            spec = lines[n][i:j - 1]
            if spec.endswith("@{}"):
                lines[n] = lines[n][:i] + spec[:-3] + "@{\\hspace{\\tabcolsep}}" + lines[n][j - 1:]
    return "\n".join(lines)



def tint_fragment(text, scope=None, set_marks=True):
    """tint_columns for a fragment of table rows (no tabular environment around it)."""
    head, tail = "\\begin{tabular}{l}\n\\midrule\n", "\n\\end{tabular}"
    out = tint_columns(head + text.rstrip("\n") + tail, scope, set_marks)
    return out[len(head):-len(tail)] + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]; scope = None
    if args[:1] == ["--scope"]:
        scope, args = args[1], args[2:]
    for f in args:
        p = Path(f); s = p.read_text(); t = tint_columns(s, scope)
        if t != s:
            p.write_text(t); print("tinted per column:", f)
        else:
            print("unchanged:", f)
