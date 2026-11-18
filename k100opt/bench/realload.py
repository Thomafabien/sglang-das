#!/usr/bin/env python3
"""真实负载压测（技术方案 §5.8）：混合长度、泊松错峰到达、真实文本、自然结束，报吞吐与 TTFT / 端到端的均值与 P50/P90/P99。

与 45 测点压测的区别：45 测点每点输入等长、请求同时到达、续写的是随机 token 文本、只报平均值；这里衡量的是
「系统吞吐」与「用户实际等待」，用来检验调度类措施和按评测分布整定的参数在真实工况下是否仍有收益。

负载（固定种子，各臂完全相同）：
  think   35%  math500 题目，对话模板、思考模式开（默认），max_new_tokens 2048          → 输入短、生成长
  qa      35%  已解答的 gsm8k 例题垫到 2048/4096/8192 token 再问一道新题，max 512          → 中等输入
  doc     20%  同上垫到 16384/32768                                                            → 长输入
  long    10%  同上垫到 65536（7%）/131072（3%）                                               → 超长输入
到达间隔为单位速率的指数分布，运行时乘 1/RATE；各请求的垫料互不相同（前缀缓存本就关闭）。

用法：
  realload.py build  <负载.jsonl> [N=120]
  realload.py run    <负载.jsonl> <输出目录> <阶段列表>     阶段列表形如 A,LAoff,A2（见 PHASES）；环境变量 RATE（请求/秒，默认 0.4）、HOST、PORT
  realload.py cmp    <输出目录> <阶段a> <阶段b>             逐项对比两个阶段
每个阶段结束打印一行以空白开头、含「行」的汇总（供看门狗 wd.sh 计数）。"""
import os, sys, json, random, time, asyncio, statistics as st

MODEL = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3.6-35B-A3B")
DS = "/workspace/datasets"
URL = "http://%s:%s/generate" % (os.environ.get("HOST", "127.0.0.1"), os.environ.get("PORT", "8023"))

# 阶段 → 运行时开关文件（服务与压测在同一容器，/tmp 共享）。只列能在运行时切换的项；需要重启的项另起实例、阶段名用 A。
PHASES = {
    "A": {}, "A2": {}, "A3": {},
    "LAoff": {"/tmp/long_admit_k": "1000"},          # 长请求准入关闭（K ≥ 1000 视为关闭）
    "LA32k": {"/tmp/long_admit_thresh": "32768"},    # 准入阈值 32768
    "ACoff": {"/tmp/achunk_off": "1"},               # 按长度分块关闭
}
SWITCH_FILES = sorted({f for v in PHASES.values() for f in v})


def build(path, n):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    rnd = random.Random(20260924)
    gsm = [json.loads(l) for l in open(f"{DS}/gsm8k_test.jsonl")]
    m500 = [json.loads(l) for l in open(f"{DS}/math500_test.jsonl")]
    ex = lambda r: "Question: " + r["question"] + "\nAnswer: " + r["answer"] + "\n\n"
    avg = sum(len(tk(ex(r)).input_ids) for r in gsm[:100]) / 100
    classes = [("think", None, 0.35), ("qa", 2048, 0.35 / 3), ("qa", 4096, 0.35 / 3), ("qa", 8192, 0.35 / 3),
               ("doc", 16384, 0.10), ("doc", 32768, 0.10), ("long", 65536, 0.07), ("long", 131072, 0.03)]
    out = []
    for i in range(n):
        c, tgt, _ = rnd.choices(classes, weights=[w for *_, w in classes])[0]
        gap = rnd.expovariate(1.0)
        if c == "think":
            q = m500[rnd.randrange(len(m500))]["problem"]
            text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            sp = {"max_new_tokens": 2048, "temperature": 0}
        else:
            qi = rnd.randrange(len(gsm))
            pool = [j for j in rnd.sample(range(len(gsm)), min(len(gsm), int(tgt / avg) + 20)) if j != qi]
            body, ntok = [], 0
            for j in pool:                                 # 逐条累加到目标长度（按平均估计，服务端报实际长度）
                if ntok + avg > tgt: break
                body.append(ex(gsm[j])); ntok += avg
            while ntok + avg <= tgt:                       # 131072 需要的例题多于 gsm8k 总量时循环复用
                body.append(ex(gsm[rnd.randrange(len(gsm))])); ntok += avg
            text = "".join(body) + "Question: " + gsm[qi]["question"] + "\nAnswer: <think>\n\n</think>\n\n"
            sp = {"max_new_tokens": 512, "temperature": 0, "stop": ["Question", "Assistant:"]}
        out.append({"i": i, "cls": c, "tgt": tgt, "gap": gap, "text": text, "sp": sp})
    with open(path, "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = {}
    for r in out: cnt[(r["cls"], r["tgt"])] = cnt.get((r["cls"], r["tgt"]), 0) + 1
    print(f"负载 {n} 条 → {path}：" + "  ".join(f"{c}{'' if t is None else t}×{k}" for (c, t), k in sorted(cnt.items(), key=lambda x: (x[0][0], x[0][1] or 0))))


async def one(sess, r, t_send, rec):
    import aiohttp
    await asyncio.sleep(max(0.0, t_send - time.perf_counter()))
    t0 = time.perf_counter(); ttft = None; meta = {}; err = None
    for attempt in range(2):          # 复用的空闲连接恰被服务端关闭时会立刻断开（未收到任何数据），换新连接重发一次
        try:
            async with sess.post(URL, json={"text": r["text"], "sampling_params": r["sp"], "stream": True},
                                 timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                async for line in resp.content:
                    line = line.strip()
                    if not line.startswith(b"data:"): continue
                    p = line[5:].strip()
                    if p == b"[DONE]": break
                    d = json.loads(p)
                    if ttft is None and d.get("text"): ttft = time.perf_counter() - t0   # 用户看到第一个字的时刻
                    meta = d.get("meta_info", meta)
            err = None; break
        except Exception as e:
            err = repr(e)[:200]
            if meta or time.perf_counter() - t0 > 1.0: break
    t1 = time.perf_counter()
    rec.append({"i": r["i"], "cls": r["cls"], "tgt": r["tgt"], "send": t0, "ttft": ttft, "e2e": t1 - t0, "end": t1,
                "in": meta.get("prompt_tokens"), "out": meta.get("completion_tokens"),
                "fin": (meta.get("finish_reason") or {}).get("type"), "err": err})


def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else float("nan")


def summarize(rec):
    ok = [x for x in rec if x["err"] is None and x["ttft"] is not None and x["out"]]
    t0 = min(x["send"] for x in rec); t1 = max(x["end"] for x in rec)
    span = t1 - t0; out = sum(x["out"] for x in ok); inp = sum(x["in"] or 0 for x in ok)
    tpot = [(x["e2e"] - x["ttft"]) / (x["out"] - 1) for x in ok if x["out"] > 1]
    s = {"n": len(rec), "ok": len(ok), "span": span, "out_tok": out, "in_tok": inp,
         "thr_out": out / span, "thr_all": (out + inp) / span}
    for k, v in (("ttft", [x["ttft"] for x in ok]), ("e2e", [x["e2e"] for x in ok]), ("tpot", tpot)):
        s[k] = {"mean": st.mean(v), "p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99)} if v else {}
    s["by_cls"] = {}
    for c in ("think", "qa", "doc", "long"):
        g = [x for x in ok if x["cls"] == c]
        if g:
            s["by_cls"][c] = {"n": len(g), "ttft_mean": st.mean(x["ttft"] for x in g), "ttft_p90": pct([x["ttft"] for x in g], 90),
                              "e2e_mean": st.mean(x["e2e"] for x in g), "e2e_p90": pct([x["e2e"] for x in g], 90)}
    return s


def line(name, s):
    return (f"  行 {name:6s} 完成 {s['ok']}/{s['n']}  用时 {s['span']:6.1f}s  输出吞吐 {s['thr_out']:7.1f} 总吞吐 {s['thr_all']:7.0f} tok/s  "
            f"TTFT 均值/P50/P90/P99 {s['ttft']['mean']:.2f}/{s['ttft']['p50']:.2f}/{s['ttft']['p90']:.2f}/{s['ttft']['p99']:.2f}s  "
            f"E2E {s['e2e']['mean']:.1f}/{s['e2e']['p50']:.1f}/{s['e2e']['p90']:.1f}/{s['e2e']['p99']:.1f}s  "
            f"TPOT 均值/P90 {s['tpot']['mean']*1e3:.1f}/{s['tpot']['p90']*1e3:.1f}ms")


async def run_phase(reqs, rate):
    import aiohttp
    rec = []; t = time.perf_counter() + 1.0
    conn = aiohttp.TCPConnector(limit=0, force_close=True)   # 每请求新连接，避免复用被关闭的空闲连接
    async with aiohttp.ClientSession(connector=conn) as sess:
        tasks = []
        for r in reqs:
            t += r["gap"] / rate
            tasks.append(asyncio.create_task(one(sess, r, t, rec)))
        await asyncio.gather(*tasks)
    return rec


def run(path, outdir, phases):
    reqs = [json.loads(l) for l in open(path)]
    rate = float(os.environ.get("RATE", "0.4"))
    os.makedirs(outdir, exist_ok=True)
    for ph in phases.split(","):
        for f in SWITCH_FILES:
            if os.path.exists(f): os.remove(f)
        for f, v in PHASES[ph].items(): open(f, "w").write(v)
        time.sleep(2.5)                                    # 运行时开关文件按 1 秒轮询
        print(f"[realload] 阶段 {ph} 开始 {time.strftime('%T')}  速率 {rate}/s  {len(reqs)} 条  开关 {PHASES[ph] or '无'}", flush=True)
        rec = asyncio.run(run_phase(reqs, rate))
        s = summarize(rec); s["phase"] = ph; s["rate"] = rate
        json.dump({"summary": s, "rec": rec}, open(f"{outdir}/{ph}.json", "w"), ensure_ascii=False)
        print(line(ph, s), flush=True)
        bad = [x for x in rec if x["err"]]
        if bad: print(f"  ⚠ {len(bad)} 条请求出错，例：{bad[0]['err']}", flush=True)
    for f in SWITCH_FILES:
        if os.path.exists(f): os.remove(f)


def cmp(outdir, a, b):
    A = json.load(open(f"{outdir}/{a}.json"))["summary"]; B = json.load(open(f"{outdir}/{b}.json"))["summary"]
    r = lambda x, y: f"{(y / x - 1) * 100:+.1f}%"
    print(f"{b} 相对 {a}：输出吞吐 {r(A['thr_out'], B['thr_out'])}  总吞吐 {r(A['thr_all'], B['thr_all'])}  用时 {r(A['span'], B['span'])}")
    for k in ("ttft", "e2e", "tpot"):
        print(f"  {k:5s} " + "  ".join(f"{q} {r(A[k][q], B[k][q])}" for q in ("mean", "p50", "p90", "p99")))
    for c in A["by_cls"]:
        if c in B["by_cls"]:
            x, y = A["by_cls"][c], B["by_cls"][c]
            print(f"  {c:5s} n={x['n']:3d}  TTFT 均值 {r(x['ttft_mean'], y['ttft_mean'])} P90 {r(x['ttft_p90'], y['ttft_p90'])}  "
                  f"E2E 均值 {r(x['e2e_mean'], y['e2e_mean'])} P90 {r(x['e2e_p90'], y['e2e_p90'])}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build": build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
    elif cmd == "run": run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "cmp": cmp(sys.argv[2], sys.argv[3], sys.argv[4])
