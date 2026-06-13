"""Fill the 3 annotator CSVs' last two columns from the submitted annotations.

Front 5 columns are kept byte-for-byte from the on-disk template (so the multi-line
model_answer fields are never re-typed); only is_hallucination/difficulty are written,
matched by seq. Asserts 160 full coverage + legal values per annotator.
"""
import csv

OUT = "human_eval"

# seq  h1 d1  h2 d2  h3 d3   (a1 | a2 | a3), transcribed row-by-row from the three files
DATA = """
H001 yes 4 yes 2 yes 2
H002 no 4 no 3 no 4
H003 yes 1 yes 1 yes 2
H004 no 1 no 1 no 1
H005 no 1 no 1 no 1
H006 yes 4 yes 5 yes 3
H007 no 1 no 1 no 1
H008 no 2 no 2 no 2
H009 no 4 no 3 no 3
H010 no 2 no 2 no 2
H011 no 2 no 2 no 2
H012 no 2 no 1 no 1
H013 yes 3 yes 3 yes 3
H014 yes 2 yes 2 yes 2
H015 yes 3 yes 3 yes 3
H016 no 1 no 2 no 2
H017 no 2 no 2 no 2
H018 no 2 no 2 no 2
H019 no 2 no 2 no 2
H020 yes 3 yes 2 yes 2
H021 yes 2 yes 2 yes 2
H022 no 4 no 2 no 2
H023 no 1 no 2 no 1
H024 yes 2 yes 2 yes 2
H025 yes 3 yes 3 yes 3
H026 no 3 no 2 no 2
H027 yes 3 yes 3 yes 3
H028 no 3 no 2 no 2
H029 no 3 no 3 no 3
H030 no 1 yes 1 yes 1
H031 yes 4 yes 4 yes 3
H032 yes 3 yes 4 yes 3
H033 yes 3 yes 4 yes 3
H034 no 4 no 2 no 2
H035 yes 2 yes 3 yes 2
H036 yes 2 yes 1 yes 2
H037 no 4 no 2 no 1
H038 yes 3 yes 2 yes 2
H039 no 2 no 2 no 2
H040 no 4 no 4 no 1
H041 no 1 no 1 no 1
H042 no 3 no 2 no 2
H043 no 2 no 2 no 2
H044 yes 4 yes 3 yes 3
H045 yes 3 yes 3 yes 4
H046 yes 3 yes 3 yes 3
H047 yes 4 yes 5 yes 3
H048 yes 4 yes 3 yes 2
H049 yes 4 yes 3 yes 4
H050 no 1 no 1 no 1
H051 no 3 no 2 no 2
H052 yes 2 yes 2 yes 2
H053 no 1 no 2 no 2
H054 no 2 no 2 no 1
H055 yes 3 yes 3 yes 3
H056 yes 3 yes 3 no 3
H057 yes 3 yes 2 yes 2
H058 no 1 no 2 no 1
H059 no 1 no 1 no 1
H060 no 2 no 1 no 1
H061 yes 3 yes 3 yes 3
H062 yes 4 yes 3 yes 3
H063 no 1 no 1 no 1
H064 yes 3 yes 3 yes 3
H065 no 1 no 1 no 1
H066 yes 2 yes 3 yes 2
H067 no 2 no 2 no 2
H068 yes 3 yes 3 yes 2
H069 no 4 no 3 no 3
H070 yes 2 yes 3 yes 2
H071 no 1 no 1 no 1
H072 no 2 no 2 no 1
H073 no 3 no 3 no 3
H074 yes 2 yes 3 yes 3
H075 no 1 no 1 no 1
H076 yes 5 yes 4 yes 4
H077 yes 4 yes 3 yes 3
H078 no 3 no 3 no 2
H079 yes 1 yes 2 yes 2
H080 no 3 no 2 no 2
H081 no 2 no 2 no 1
H082 yes 2 yes 1 yes 2
H083 yes 3 yes 3 yes 3
H084 yes 3 yes 3 yes 3
H085 no 2 no 1 no 1
H086 no 4 no 3 no 3
H087 no 4 no 4 no 4
H088 yes 3 yes 3 yes 3
H089 no 3 no 3 no 2
H090 no 3 no 3 no 2
H091 no 2 no 3 no 2
H092 no 3 no 2 no 2
H093 no 3 no 3 no 2
H094 no 2 no 1 no 1
H095 no 2 no 2 no 2
H096 yes 3 yes 3 yes 3
H097 no 4 no 2 no 1
H098 no 2 no 2 no 1
H099 yes 4 yes 3 yes 3
H100 yes 3 yes 2 yes 3
H101 no 3 no 4 no 3
H102 no 2 no 3 no 2
H103 no 2 no 2 no 2
H104 yes 3 yes 3 yes 3
H105 no 2 no 2 no 2
H106 no 1 no 2 no 1
H107 no 1 no 2 no 1
H108 no 1 no 1 no 1
H109 yes 2 yes 2 yes 2
H110 yes 4 yes 4 yes 4
H111 no 2 no 2 no 2
H112 no 1 no 1 no 1
H113 yes 2 no 2 no 2
H114 no 3 no 3 no 2
H115 yes 2 yes 2 yes 2
H116 no 3 no 3 no 2
H117 yes 2 yes 2 yes 2
H118 no 3 no 2 no 1
H119 no 2 no 2 no 2
H120 no 2 no 2 no 2
H121 no 3 no 2 no 3
H122 no 2 no 2 no 2
H123 no 2 no 1 no 1
H124 no 4 no 5 no 3
H125 yes 3 yes 2 yes 2
H126 no 2 no 2 no 2
H127 no 1 no 1 no 1
H128 no 1 no 2 no 2
H129 yes 3 yes 2 yes 2
H130 no 1 no 2 no 2
H131 no 1 no 1 no 1
H132 no 1 no 1 no 1
H133 yes 2 yes 3 yes 3
H134 yes 3 yes 3 yes 3
H135 no 3 no 3 no 3
H136 no 1 no 2 no 2
H137 yes 3 yes 3 yes 3
H138 no 2 no 2 no 2
H139 no 3 no 2 no 2
H140 yes 2 yes 2 yes 2
H141 no 3 no 2 no 3
H142 yes 3 yes 3 yes 4
H143 yes 3 yes 2 yes 3
H144 yes 3 yes 3 yes 3
H145 no 3 no 3 no 1
H146 no 3 no 3 no 3
H147 no 1 no 1 no 1
H148 no 3 yes 3 no 2
H149 no 2 no 2 no 2
H150 no 3 no 3 no 3
H151 yes 3 yes 2 yes 2
H152 yes 4 yes 5 yes 3
H153 no 4 no 2 no 2
H154 no 1 no 1 no 1
H155 no 1 no 2 no 2
H156 yes 4 yes 4 yes 5
H157 no 2 no 1 no 2
H158 no 2 no 2 no 2
H159 yes 3 yes 3 yes 3
H160 no 2 no 2 no 2
"""

ann = {1: {}, 2: {}, 3: {}}
for line in DATA.strip().splitlines():
    p = line.split()
    seq = p[0]
    ann[1][seq] = (p[1], p[2])
    ann[2][seq] = (p[3], p[4])
    ann[3][seq] = (p[5], p[6])

# validate transcription
for a in (1, 2, 3):
    assert len(ann[a]) == 160, f"annotator {a}: {len(ann[a])} seqs (expected 160)"
    for seq, (h, d) in ann[a].items():
        assert h in ("yes", "no"), f"{a}/{seq}: bad h={h!r}"
        assert d in ("1", "2", "3", "4", "5"), f"{a}/{seq}: bad d={d!r}"
print("transcription OK: 3 annotators x 160 items, all legal")

# write back, keeping front 5 columns from disk untouched
for a in (1, 2, 3):
    path = f"{OUT}/annotator_{a}.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    filled = 0
    for r in rows:
        seq = r["seq"]
        if seq in ann[a]:
            r["is_hallucination(yes/no)"], r["difficulty(1-5)"] = ann[a][seq]
            filled += 1
    assert filled == 160, f"annotator {a}: filled {filled} (expected 160)"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"annotator_{a}.csv: filled {filled} rows")
