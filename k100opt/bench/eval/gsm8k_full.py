#!/usr/bin/env python3
"""gsm8k 全量（1319 题，5-shot 补全式，强制空思考块，temperature 0）。用法: gsm8k_full.py [题数] [线程数]"""
import os, sys, json, re, ast, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1319; TH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
rows = [json.loads(l) for l in open("/workspace/datasets/gsm8k_test.jsonl")]
def one(i, inc):
    r = "Question: " + rows[i]["question"] + "\nAnswer:"
    return r + " " + rows[i]["answer"] if inc else r
PRE = "".join(one(i, True) + "\n\n" for i in range(5))
def gold(a): return ast.literal_eval(a.split("####")[-1].strip().replace(",", ""))
def ask(i):
    body = {"text": PRE + one(i, False) + " <think>\n\n</think>\n\n",
            "sampling_params": {"max_new_tokens": 512, "temperature": 0, "stop": ["Question", "Assistant:"]}}
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://%s:%s/generate" % (os.environ.get("HOST", "127.0.0.1"), os.environ.get("PORT", "8023")), data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}), timeout=600)); break
        except Exception: time.sleep(2)
    else: return None, "err"
    n = re.findall(r"-?\d+\.?\d*", r["text"].replace(",", ""))
    try: p = ast.literal_eval(n[-1]) if n else None
    except Exception: p = None
    return p, r.get("meta_info", {}).get("finish_reason", {}).get("type")
Q = list(range(5, min(len(rows), 5 + N))); t0 = time.time()
with ThreadPoolExecutor(TH) as ex: got = list(ex.map(ask, Q))
ok = sum(1 for i, (p, _) in zip(Q, got) if p is not None and abs(p - gold(rows[i]["answer"])) < 1e-6)
trunc = sum(1 for _, f in got if f == "length"); err = sum(1 for _, f in got if f == "err")
print(f"gsm8k: 正确 {ok}/{len(Q)} = {100*ok/len(Q):.2f}%   截断 {trunc}  请求失败 {err}   用时 {time.time()-t0:.0f}s")
