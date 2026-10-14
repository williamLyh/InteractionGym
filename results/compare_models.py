"""Two duplex agents on the open- vs closed-loop benchmarks, paired per (item, seed): each model's A and B, the
Gander - MiniCPM-o difference in each condition, and how much the closed loop changes that gap
((gB - mB) - (gA - mA)), with 95% bootstrap CIs. FD-Bench v2 on seeds 0-1 (Gander was not run on seed 2).

    python results/compare_models.py <minicpmo runs> <gander runs>   # folders holding fdb3_ab/, fdb2_ab/, audiomc_ab/
"""
import json
import random
import sys
from pathlib import Path

MO, GA = Path(sys.argv[1]), Path(sys.argv[2])
random.seed(0)
def load(p): return json.load(open(p))
def boot(pairs, B=4000):
    # pairs: list of (mo_val, ga_val); returns mean diff ga-mo and CI
    n=len(pairs); d=[g-m for m,g in pairs]; mean=sum(d)/n
    bs=sorted(sum(d[random.randrange(n)] for _ in range(n))/n for _ in range(B))
    return mean, bs[int(.025*B)], bs[int(.975*B)]
def boot_dd(items, B=4000):
    # items: (moA, moB, gaA, gaB); returns (gaB-moB)-(gaA-moA)
    n=len(items); d=[(gb-mb)-(ga-ma) for ma,mb,ga,gb in items]; mean=sum(d)/n
    bs=sorted(sum(d[random.randrange(n)] for _ in range(n))/n for _ in range(B))
    return mean, bs[int(.025*B)], bs[int(.975*B)]
def report(name, mo, ga):  # dicts key -> (A, B)
    keys=sorted(set(mo)&set(ga)); n=len(keys)
    mA=sum(mo[k][0] for k in keys)/n; mB=sum(mo[k][1] for k in keys)/n
    gA=sum(ga[k][0] for k in keys)/n; gB=sum(ga[k][1] for k in keys)/n
    a=boot([(mo[k][0],ga[k][0]) for k in keys]); b=boot([(mo[k][1],ga[k][1]) for k in keys])
    dd=boot_dd([(mo[k][0],mo[k][1],ga[k][0],ga[k][1]) for k in keys])
    f=lambda t:f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    print(f"| {name} | {n} | {mA:.3f} | {mB:.3f} | {gA:.3f} | {gB:.3f} | {f(a)} | {f(b)} | {f(dd)} |")
print("| metric | n | MiniCPM-o A | MiniCPM-o B | Gander A | Gander B | Gander−MiniCPM-o, A | Gander−MiniCPM-o, B | gap change (B−A) |")
print("|---|---|---|---|---|---|---|---|---|")
# FDB-v3
def f3(p):
    out={}
    for r in load(p):
        out[(r['recording'],r['seed'])]=r
    return out
m3=f3(MO / 'fdb3_ab/minicpmo/rows.json'); g3=f3(GA / 'fdb3_ab/gander/rows.json')
report("FDB-v3 spoken fulfilment (A vs B final)", {k:(float(r['A']['spoken']['fulfilled']),float(r['B']['spoken_final']['fulfilled'])) for k,r in m3.items()},
       {k:(float(r['A']['spoken']['fulfilled']),float(r['B']['spoken_final']['fulfilled'])) for k,r in g3.items()})
report("FDB-v3 slot outcome", {k:(r['A']['spoken']['outcome'],r['B']['spoken_final']['outcome']) for k,r in m3.items()},
       {k:(r['A']['spoken']['outcome'],r['B']['spoken_final']['outcome']) for k,r in g3.items()})
report("FDB-v3 response quality", {k:(r['A']['response_quality'],r['B']['response_quality']) for k,r in m3.items() if r['A']['response_quality'] is not None and r['B']['response_quality'] is not None},
       {k:(r['A']['response_quality'],r['B']['response_quality']) for k,r in g3.items() if r['A']['response_quality'] is not None and r['B']['response_quality'] is not None})
# FDB-v2
def f2(p, seeds=(0,1)):
    d={}
    for r in load(p):
        if r['seed'] in seeds: d.setdefault((r['task_id'],r['seed']),{})[r['cond']]=r
    return {k:v for k,v in d.items() if 'A' in v and 'B' in v}
m2=f2(MO / 'fdb2_ab/rows_180.json'); g2=f2(GA / 'fdb2_ab/rows_180.json')
for met in ('if','tt','task'):
    sel=lambda d:{k:(v['A'][met],v['B'][met]) for k,v in d.items() if v['A'][met] is not None and v['B'][met] is not None}
    report(f"FDB-v2 {met.upper()} (seeds 0-1, B 180 s)", sel(m2), sel(g2))
# AMC
def fa(p):
    d={}
    for r in load(p): d.setdefault((r['task_id'],r['seed']),{})[r['cond']]=r
    return {k:v for k,v in d.items() if 'A' in v and 'B' in v}
ma=fa(MO / 'audiomc_ab/rows.json'); gaa=fa(GA / 'audiomc_ab/rows.json')
report("AudioMC rubric items passed", {k:(v['A']['rubric_rate'],v['B']['rubric_rate']) for k,v in ma.items()}, {k:(v['A']['rubric_rate'],v['B']['rubric_rate']) for k,v in gaa.items()})
report("AudioMC all items passed", {k:(float(v['A']['all_pass']),float(v['B']['all_pass'])) for k,v in ma.items()}, {k:(float(v['A']['all_pass']),float(v['B']['all_pass'])) for k,v in gaa.items()})
