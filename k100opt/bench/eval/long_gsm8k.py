#!/usr/bin/env python3
"""长上下文 + 并发下的答案正确性：用大量已解答的 gsm8k 例题把提示词垫到指定长度（多块预填充），再问一道新题，判最终数值。
用法: long_gsm8k.py <目标token数> <题数> <并发>    同一脚本分别在基线实例与最终配置实例上跑，对比正确率。"""
import os, sys, json, re, ast, time, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
TOK, N, TH = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tk = AutoTokenizer.from_pretrained(os.environ.get("MODEL_PATH", "/workspace/models/Qwen3.6-35B-A3B"))
rows = [json.loads(l) for l in open("/workspace/datasets/gsm8k_test.jsonl")]
def ex(i): return "Question: " + rows[i]["question"] + "\nAnswer: " + rows[i]["answer"] + "\n\n"
def gold(a): return ast.literal_eval(a.split("####")[-1].strip().replace(",", ""))
Q = list(range(0, N)); pool = list(range(N, len(rows)))
avg = sum(len(tk(ex(i)).input_ids) for i in pool[:50]) / 50; shots = max(5, int(TOK / avg))
def build(i):
    rnd = random.Random(1000 + i); ids = rnd.sample(pool, min(shots, len(pool)))     # 每题的垫料不同，避免任何前缀复用
    return "".join(ex(j) for j in ids) + "Question: " + rows[i]["question"] + "\nAnswer: <think>\n\n</think>\n\n"
def ask(i):
    body = {"text": build(i), "sampling_params": {"max_new_tokens": 512, "temperature": 0, "stop": ["Question", "Assistant:"]}}
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://%s:%s/generate" % (os.environ.get("HOST", "127.0.0.1"), os.environ.get("PORT", "8023")), data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}), timeout=1800)); break
        except Exception as e: time.sleep(3)
    else: return None, "err", 0
    n = re.findall(r"-?\d+\.?\d*", r["text"].replace(",", ""))
    try: p = ast.literal_eval(n[-1]) if n else None
    except Exception: p = None
    return p, r["meta_info"].get("finish_reason", {}).get("type"), r["meta_info"]["prompt_tokens"]
t0 = time.time()
with ThreadPoolExecutor(TH) as e: got = list(e.map(ask, Q))
ok = sum(1 for i, (p, _, _) in zip(Q, got) if p is not None and abs(p - gold(rows[i]["answer"])) < 1e-6)
pt = [g[2] for g in got if g[2]]
print(f"长上下文 gsm8k: 目标 {TOK} token（实际 {min(pt)}–{max(pt)}，{shots} 个例题） 并发 {TH}: 正确 {ok}/{len(Q)} = {100*ok/len(Q):.1f}%   截断 {sum(1 for g in got if g[1]=='length')}  失败 {sum(1 for g in got if g[1]=='err')}  用时 {time.time()-t0:.0f}s")
