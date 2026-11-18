#!/usr/bin/env python3
"""45 测点 × 3 遍全量测量（与 BASELINE.md 中分数同一口径）。

用法（必须带 PYTHONPATH，原因见 bench/README.md）：
    cd <repo>/bench
    PYTHONPATH=/workspace/sglang-das/python python3 run45.py
    PYTHONPATH=/workspace/sglang-das/python python3 score45.py

可用环境变量：
    MODEL_PATH   模型路径，默认 /workspace/models/Qwen3.6-35B-A3B
    HOST / PORT  服务地址，默认 127.0.0.1 / 8023
    OUTDIR       输出目录，默认 ./out
    PASSES       遍数，默认 3（正式评分口径为 3 遍全平均；粗测可设 1）
    ROWS         逗号分隔的行号，只测这些测点（抽测用）；默认全部 45 行
"""
import sys, os, json, io, re, time, contextlib, runpy

HERE  = os.path.dirname(os.path.abspath(__file__))
SPEC  = json.load(open(os.path.join(HERE, "spec45.json")))
MODEL = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3.6-35B-A3B")
HOST  = os.environ.get("HOST", "127.0.0.1")
PORT  = os.environ.get("PORT", "8023")
OUTDIR= os.environ.get("OUTDIR", os.path.join(HERE, "out"))
PASSES= int(os.environ.get("PASSES", "3"))
ROWS  = [int(x) for x in os.environ.get("ROWS", "").replace(" ", "").split(",") if x]   # 只测这些行（抽测用）；空 = 全部 45 行
if ROWS:
    SPEC = [s for s in SPEC if s["excel_row"] in ROWS]
RES   = os.path.join(OUTDIR, "results")
OUT   = os.path.join(OUTDIR, "formal_results.json")
os.makedirs(RES, exist_ok=True)

# 启动前自检：确认用的是 sglang-das 而不是 pip 装的旧版
try:
    import sglang, sglang.benchmark.serving   # noqa: F401
    src = os.path.dirname(os.path.dirname(sglang.__file__))
    print(f"[自检] sglang 来源: {src}")
    if "sglang-das" not in src and "site-packages" in src:
        print("[自检] !! 当前用的是 pip 安装的 sglang，不是 sglang-das。")
        print("[自检] !! 该版本的随机种子与数据集实现不同，测出的数字与基线不可比。")
        print("[自检] !! 请带上 PYTHONPATH=/workspace/sglang-das/python 重跑。")
        sys.exit(2)
except ImportError as e:
    print(f"[自检] 无法导入 sglang.benchmark.serving：{e}")
    print("[自检] 请确认 PYTHONPATH 指向 sglang-das/python。")
    sys.exit(2)


def bench(tag, inp, out, conc):
    """单个测点。注意：不传 --warmup-requests，使用其默认值 1（热口径）。"""
    argv = ["b", "--backend", "sglang", "--host", HOST, "--port", PORT,
            "--dataset-name", "random-ids",
            "--num-prompts", str(conc), "--max-concurrency", str(conc),
            "--random-input-len", str(inp), "--random-output-len", str(out),
            "--random-range-ratio", "1", "--request-rate", "inf", "--disable-tqdm",
            "--output-details", "--output-file", os.path.join(RES, tag + ".jsonl"),
            "--tokenizer", MODEL]
    sys.argv = argv
    buf = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_module("sglang.benchmark.serving", run_name="__main__")
        except SystemExit:
            pass
    s = buf.getvalue()
    g = lambda p, d="0": (re.search(p, s).group(1) if re.search(p, s) else d)
    acc = re.findall(r"(?i)accept length:\s*([\d.]+)", s)
    return dict(
        successful      = int(g(r'Successful requests:\s+(\d+)')),
        duration_s      = float(g(r'Benchmark duration \(s\):\s+([\d.]+)')),
        req_thr         = float(g(r'Request throughput \(req/s\):\s+([\d.]+)')),
        in_tok_thr      = float(g(r'Input token throughput \(tok/s\):\s+([\d.]+)')),
        out_tok_thr     = float(g(r'Output token throughput \(tok/s\):\s+([\d.]+)')),
        peak_out_thr    = float(g(r'Peak output token throughput \(tok/s\):\s+([\d.]+)')),
        total_tok_thr   = float(g(r'Total token throughput \(tok/s\):\s+([\d.]+)')),
        mean_ttft       = float(g(r'Mean TTFT \(ms\):\s+([\d.]+)')),
        mean_tpot       = float(g(r'Mean TPOT \(ms\):\s+([\d.]+)')),
        accept          = float(acc[-1]) if acc else float("nan"),
        wall_s          = time.time() - t0,
    )


allres = {}
t_start = time.time()
for p in range(1, PASSES + 1):
    print(f"\n=== 第 {p} 遍（共 {PASSES} 遍）===", flush=True)
    for s in SPEC:
        r = s["excel_row"]
        tag = "p%d_row%02d_in%d_out%d_c%d" % (p, r, s["inp"], s["out"], s["conc"])
        m = bench(tag, s["inp"], s["out"], s["conc"])
        allres.setdefault(str(r), []).append(m)
        json.dump(allres, open(OUT, "w"), indent=1)
        print("  行%2d in=%-6d out=%-5d c=%-2d | TTFT=%10.2f TPOT=%7.3f ok=%d/%d acc=%.2f %5.0fs"
              % (r, s["inp"], s["out"], s["conc"], m["mean_ttft"], m["mean_tpot"],
                 m["successful"], s["conc"], m["accept"], m["wall_s"]), flush=True)
    print("  -- 第 %d 遍完成，累计 %.1f 分钟 --" % (p, (time.time() - t_start) / 60), flush=True)

print(f"\n全部完成，总用时 {(time.time()-t_start)/60:.1f} 分钟")
print(f"结果：{OUT}\n下一步：PYTHONPATH=... python3 score45.py")
