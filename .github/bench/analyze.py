#!/usr/bin/env python3
"""Summarise run.sh outputs (one or more replica dirs) as markdown. Bench branch only."""
import collections, glob, os, re, statistics as st, sys

root = sys.argv[1]
CFGS = ["base", "head-off", "head-on", "head-on-nostats", "head-on-rs"]
LINE = re.compile(r"Query (q\d+) iteration (\d+) took ([\d.]+) ms.*returned (\d+) rows")
TAG = re.compile(r"(focused|full)-w(\d+)-([a-z-]+)-r(\d+)\.(log|perf|time)$")

# times[(phase,w,cfg,query)] -> list of per-run medians (one per round x replica)
times = collections.defaultdict(list)
totals = collections.defaultdict(list)  # (phase,w,cfg) -> per-run sum of query medians
rowvals = collections.defaultdict(lambda: collections.defaultdict(set))
perf = collections.defaultdict(lambda: collections.defaultdict(list))  # (w,cfg)->event->[vals]
failures = []

for f in sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True)):
    m = TAG.search(os.path.basename(f))
    if not m:
        if f.endswith("failures.txt"):
            failures += open(f).read().split("\n")
        continue
    phase, w, cfg, r, ext = m.groups()
    w = int(w)
    if ext == "log":
        per = collections.defaultdict(list)
        for l in open(f, errors="replace"):
            mm = LINE.search(l)
            if mm:
                q, it, ms, n = mm[1], int(mm[2]), float(mm[3]), int(mm[4])
                rowvals[q][cfg].add(n)
                per[q].append((it, ms))
        tot = 0.0
        for q, v in per.items():
            vals = [ms for it, ms in v if it > 0] or [ms for _, ms in v]  # drop warm-up iteration 0
            med = st.median(vals)
            times[(phase, w, cfg, q)].append(med)
            tot += med
        if per:
            totals[(phase, w, cfg)].append(tot)
    elif ext == "perf":
        for l in open(f):
            p = l.strip().split(",")
            if len(p) >= 3 and p[0] and p[0][0].isdigit():
                perf[(w, cfg)][p[2].split(":")[0]].append(float(p[0]))
    elif ext == "time":
        txt = open(f).read()
        u = re.search(r"User time \(seconds\): ([\d.]+)", txt)
        s = re.search(r"System time \(seconds\): ([\d.]+)", txt)
        if u and s:
            perf[(w, cfg)]["user+sys-s"].append(float(u[1]) + float(s[1]))


def fmt(v):
    if not v:
        return "–"
    return f"{st.median(v):.0f} [{min(v):.0f}–{max(v):.0f}] n={len(v)}"


def pct(v, ref):
    if not v or not ref:
        return "–"
    return f"{100 * (st.median(v) / st.median(ref) - 1):+.1f}%"


out = []
cfgs = [c for c in CFGS if any(k[2] == c for k in totals)]
for phase in ["focused", "full"]:
    for w in [0, 4]:
        qs = sorted({k[3] for k in times if k[0] == phase and k[1] == w}, key=lambda x: int(x[1:]))
        if not qs:
            continue
        out.append(f"\n### {phase} — workers {w}\n")
        out.append("median ms [min–max of per-run medians], % vs head-off\n")
        out.append("| query | " + " | ".join(cfgs) + " |")
        out.append("|---" * (len(cfgs) + 1) + "|")
        for q in qs + ["TOTAL"]:
            cells = []
            for c in cfgs:
                if q == "TOTAL":
                    v, ref = totals[(phase, w, c)], totals[(phase, w, "head-off")]
                else:
                    v, ref = times[(phase, w, c, q)], times[(phase, w, "head-off", q)]
                cells.append(f"{fmt(v)} ({pct(v, ref)})")
            out.append(f"| {q} | " + " | ".join(cells) + " |")

if perf:
    out.append("\n### CPU work per focused run (median across runs, % vs head-off)\n")
    events = sorted({e for d in perf.values() for e in d})
    out.append("| workers | config | " + " | ".join(events) + " |")
    out.append("|---" * (len(events) + 2) + "|")
    for w in [0, 4]:
        for c in cfgs:
            if (w, c) not in perf:
                continue
            cells = []
            for e in events:
                v, ref = perf[(w, c)].get(e, []), perf[(w, "head-off")].get(e, [])
                cells.append(f"{st.median(v):.4g} ({pct(v, ref)})" if v else "–")
            out.append(f"| {w} | {c} | " + " | ".join(cells) + " |")

out.append("\n### Correctness (row counts per query across all configs/workers/runs)\n")
bad = [q for q, d in rowvals.items() if len(set().union(*d.values())) > 1]
if bad:
    for q in sorted(bad, key=lambda x: int(x[1:])):
        out.append(f"- MISMATCH {q}: " + ", ".join(f"{c}={sorted(v)}" for c, v in rowvals[q].items()))
else:
    out.append(f"All {len(rowvals)} queries returned identical row counts in every config.")
failures = [f for f in failures if f.strip()]
if failures:
    out.append("\n### Failures\n" + "\n".join(f"- {f}" for f in failures))
print("\n".join(out))
