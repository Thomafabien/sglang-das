#!/usr/bin/env python3
"""长上下文哨兵：同一条提示在不同量化配置下的接受长度与输出是否一致。
用法: sentinel.py <real|rand> <token数> <配置...>   配置形如 off / G128 / G32"""
import sys, os, json, time, re, urllib.request, random
kind, N, cfgs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
LOG = os.environ.get("SRVLOG", "/workspace/k100opt/bench/out/server.log")
if kind == "real":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("MODEL_PATH", "/workspace/models/Qwen3.6-35B-A3B"))
    rows = [json.loads(l) for l in open("/workspace/datasets/gsm8k_test.jsonl")]
    txt = "".join("Question: " + r["question"] + "\nAnswer: " + r["answer"] + "\n\n" for r in rows)
    ids = tok(txt).input_ids
    while len(ids) < N: ids = ids + ids
    ids = ids[:N - 40] + tok("Question: " + rows[7]["question"] + "\nAnswer:").input_ids
else:
    random.seed(int(os.environ.get("SEED","42"))); ids = [random.randint(1000, 150000) for _ in range(N)]
def run():
    n0 = sum(1 for _ in open(LOG, errors="replace"))
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://%s:%s/generate" % (os.environ.get("HOST", "127.0.0.1"), os.environ.get("PORT", "8023")),
        data=json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 256, "temperature": 0, "ignore_eos": True}}).encode(),
        headers={"Content-Type": "application/json"}), timeout=1200))
    time.sleep(1)
    acc = [float(m.group(1)) for l in list(open(LOG, errors="replace"))[n0:] if "TP0" in l for m in [re.search(r"accept len: ([\d.]+)", l)] if m]
    mi = r["meta_info"]
    return r["text"], acc, mi.get("spec_verify_ct"), mi.get("completion_tokens")
base = None
for c in cfgs:
    if c == "off": open("/tmp/qar_off", "w").close()
    else:
        if os.path.exists("/tmp/qar_off"): os.remove("/tmp/qar_off")
        json.dump({"G": int(c[1:]), "debug": 0}, open("/tmp/qar_cfg.json", "w"))
    time.sleep(1.5)
    out = []
    for k in range(int(os.environ.get("REP","2"))):
        t, acc, vct, ntok = run()
        al = (ntok / vct) if vct else float("nan")
        out.append((t, al, acc))
    if base is None: base = out[-1][0]
    same = sum(1 for a, b in zip(out[-1][0], base) if a == b)
    als = " / ".join("%.3f" % o[1] for o in out)
    print(f"{kind} {len(ids)} tok  配置 {c:<5} 接受长度(完成数/验证次数) {als}   日志 {out[-1][2]}   与首配置输出的公共前缀字符 {os.path.commonprefix([out[-1][0], base]).__len__()}/{len(base)}", flush=True)
if os.path.exists("/tmp/qar_off"): os.remove("/tmp/qar_off")
