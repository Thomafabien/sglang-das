#!/usr/bin/env python3
"""对 run45.py 的结果评分，并与本仓库的基准值逐行对照。正式口径：45 测点 × 3 遍全平均（不丢首遍）；也可用于抽测与单遍。

    O = mean( L20_TTFT / 0.7 / 本平台_TTFT ) × 100
    P = mean( L20_TPOT / 0.7 / 本平台_TPOT ) × 100
"""
import json, os, statistics as st
from collections import defaultdict

HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("OUTDIR", os.path.join(HERE, "out"))
SPEC   = {s["excel_row"]: s for s in json.load(open(os.path.join(HERE, "spec45.json")))}
R      = json.load(open(os.path.join(OUTDIR, "formal_results.json")))
# 逐行对照的基准按硬件档位选取：液冷 baseline.json（R21 单遍）；风冷（SGLANG_HW_PROFILE=air）baseline_air.json
# （R21 + 风冷档位，交付方式单遍，由 docs/结果数据/风冷_R21交付方式_单遍_4-7卡_formal_results.json 生成）
AIR    = os.environ.get("SGLANG_HW_PROFILE", "") == "air"
BASE_F = os.environ.get("BASELINE") or os.path.join(HERE, "baseline_air.json" if AIR else "baseline.json")
BASE   = json.load(open(BASE_F))

rows = []
for k in sorted(R, key=int):
    r = int(k); reps = R[k]; s = SPEC[r]
    npass = len(reps)
    f = lambda key: [x[key] for x in reps]
    ttft = f("mean_ttft"); tpot = f("mean_tpot")
    m_t = sum(ttft) / len(ttft); m_p = sum(tpot) / len(tpot)
    rows.append(dict(
        row=r, inp=s["inp"], out=s["out"], conc=s["conc"],
        ttft=m_t, tpot=m_p,
        cv_ttft=100 * (st.stdev(ttft) if len(ttft) > 1 else 0) / m_t,
        cv_tpot=100 * (st.stdev(tpot) if len(tpot) > 1 else 0) / m_p,
        O=s["l20_ttft"] / 0.7 / m_t * 100,
        P=s["l20_tpot"] / 0.7 / m_p * 100,
        l20_ttft=s["l20_ttft"], l20_tpot=s["l20_tpot"],
        succ=reps[0]["successful"],
        accept=sum(f("accept")) / len(reps),
    ))
json.dump(rows, open(os.path.join(OUTDIR, "scored45.json"), "w"), indent=1, ensure_ascii=False)

def dev(x, key):
    return 100 * (x[key] / BASE[str(x["row"])][key] - 1)
def tol_ttft(x):      # 短输入测点跨实例漂移大于长输入
    return 10.0 if x["inp"] <= 4096 else 5.0
TOL_TPOT = 12.0       # TPOT 含投机解码接受长度的波动
print(f"{'行':>3} {'in':>7} {'out':>5} {'c':>3} | {'TTFT(ms)':>10} {'基准':>9} {'偏差':>7} | {'TPOT(ms)':>9} {'基准':>8} {'偏差':>7} | {'O':>7} {'P':>7}")
flags = []
for x in rows:
    dt, dp = dev(x, "ttft"), dev(x, "tpot")
    f1 = " 慢" if dt > tol_ttft(x) else ""; f2 = " 慢" if dp > TOL_TPOT else ""
    if f1 or f2: flags.append(x["row"])
    print(f"{x['row']:>3} {x['inp']:>7} {x['out']:>5} {x['conc']:>3} | "
          f"{x['ttft']:10.2f} {BASE[str(x['row'])]['ttft']:9.1f} {dt:+6.1f}%{f1} | {x['tpot']:9.3f} {BASE[str(x['row'])]['tpot']:8.3f} {dp:+6.1f}%{f2} | "
          f"{x['O']:7.2f} {x['P']:7.2f}")

O = sum(x["O"] for x in rows) / len(rows)
P = sum(x["P"] for x in rows) / len(rows)
A = sum(x["accept"] for x in rows) / len(rows)
cv = sorted(x["cv_ttft"] for x in rows)
print(f"\n{'='*72}")
bO = sum(SPEC[x["row"]]["l20_ttft"] / 0.7 / BASE[str(x["row"])]["ttft"] * 100 for x in rows) / len(rows)
bP = sum(SPEC[x["row"]]["l20_tpot"] / 0.7 / BASE[str(x["row"])]["tpot"] * 100 for x in rows) / len(rows)
full = len(rows) == 45
print(f"O = {O:.2f}    P = {P:.2f}    测点数 = {len(rows)}    遍数 = {npass}" + ("" if full else "    （抽测：仅为所测测点的均值，不是全表成绩）"))
print(f"基准文件：{os.path.basename(BASE_F)}（{'风冷档位' if AIR else '液冷'}；可用环境变量 BASELINE 指定）")
print(f"同一批测点的基准值：O = {bO:.2f}    P = {bP:.2f}    →  相对基准 O {O-bO:+.2f} / P {P-bP:+.2f}")
if npass > 1:
    print(f"{npass} 遍离散度(TTFT)：中位 {cv[len(cv)//2]:.2f}%  最大 {cv[-1]:.2f}%")
if full and npass >= 3:
    print(f"判定：O {'通过' if O > 100 else '未达标'}  /  P {'通过' if P > 100 else '未达标'}")
else:
    print("说明：正式成绩以 45 测点 × 3 遍为准；「单遍 O 比三遍低约 1 点、P 相差不到 0.3」是 R16 时期的观察，R18 修复首遍效应后未复核。")
if flags:
    print(f"偏慢的测点（TTFT 超过基准 {5}%/{10}%（长/短输入）或 TPOT 超过 {TOL_TPOT:.0f}%）：行 {flags}")
    print("  排查顺序：① 上面的装载记录是否 22 项（风冷档位 25 项）、[switches] 是否齐全；② 开测前 GPU 是否空闲；③ MoE 配置来源是否指向本仓库；④ 个别单并发测点仅 TPOT 偏差，多为投机解码接受长度的波动。")
else:
    print("全部测点在容差内。")

for label, key in (("并发度", "conc"), ("输入长度", "inp")):
    g = defaultdict(list)
    for x in rows: g[x[key]].append(x)
    print(f"\n按{label}分组：")
    for k in sorted(g):
        v = g[k]
        print(f"  {k:<8} n={len(v):<3} O={sum(y['O'] for y in v)/len(v):6.2f}  P={sum(y['P'] for y in v)/len(v):6.2f}")
