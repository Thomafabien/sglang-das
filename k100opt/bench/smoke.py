#!/usr/bin/env python3
"""抽样冒烟测试 —— 两分钟内回答两个问题：改进有没有效？答得对不对？
用法: python3 smoke.py [标签]
每个测点跑两遍：第 1 遍可能受冷形状影响，以第 2 遍与基线比对。
"""
import sys,os,json,io,re,ast,time,contextlib,runpy,urllib.request
from concurrent.futures import ThreadPoolExecutor
TAG=sys.argv[1] if len(sys.argv)>1 else time.strftime("%m%d-%H%M")
MODEL=os.environ.get("MODEL_PATH","/workspace/models/Qwen3.6-35B-A3B")
HIST=os.path.join(os.path.dirname(os.path.abspath(__file__)),"smoke_history.jsonl")
# 基线：本仓库当前版本（2026-09-20，由交付镜像创建的全新容器）在同一冷口径下的实测第 2 遍值。
# 注意本脚本传 --warmup-requests 0（冷口径），与 45 行正式表格的热口径不同，
# 故此处的数不能与 BASELINE.md 的 O/P 对照，只用于确认环境搭对了没有：
# 环境正确时三行都应落在 ±5% 内。
BASE={4:(151.2,4.210), 13:(1510.2,13.160), 29:(1335.7,4.540)}
ROWS=[(4,1024,1024,1,"短输入·单并发"),(13,2048,1024,16,"中输入·高并发"),(29,16384,1024,1,"长输入·单并发")]

def bench(inp,out,conc):
    sys.argv=["b","--backend","sglang","--host",os.environ.get("HOST","127.0.0.1"),"--port",os.environ.get("PORT","8023"),
      "--dataset-name","random-ids","--num-prompts",str(conc),"--max-concurrency",str(conc),
      "--random-input-len",str(inp),"--random-output-len",str(out),"--random-range-ratio","1",
      "--request-rate","inf","--warmup-requests","0","--disable-tqdm","--tokenizer",MODEL]
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: runpy.run_module("sglang.benchmark.serving",run_name="__main__")
        except SystemExit: pass
    s=buf.getvalue(); g=lambda p,d="0":(re.search(p,s).group(1) if re.search(p,s) else d)
    return (float(g(r'Mean TTFT \(ms\):\s+([\d.]+)')), float(g(r'Mean TPOT \(ms\):\s+([\d.]+)')),
            int(g(r'Successful requests:\s+(\d+)')))

def post(path,body,timeout=120):
    return json.load(urllib.request.urlopen(urllib.request.Request(
        "http://%s:%s"%(os.environ.get("HOST","127.0.0.1"),os.environ.get("PORT","8023"))+path,data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}),timeout=timeout))

t0=time.time()
print("="*74); print("  抽样冒烟测试  [%s]"%TAG); print("="*74)

print("\n【一】改进有没有效 —— 3 个代表性测点各跑两遍\n")
print("  %-15s %10s %10s %7s | %9s %9s %7s"%("测点","TTFT第1遍","TTFT第2遍","vs基线","TPOT第1遍","TPOT第2遍","vs基线"))
perf={}; fail=[]; cold=[]
for row,inp,out,conc,name in ROWS:
    a=bench(inp,out,conc); b=bench(inp,out,conc)
    if a[2]!=conc or b[2]!=conc: fail.append("行%d 成功数 %d/%d、%d/%d"%(row,a[2],conc,b[2],conc))
    base=BASE[row]
    dt=100*(b[0]-base[0])/base[0]; dp=100*(b[1]-base[1])/base[1]
    gap=100*(a[0]-b[0])/b[0]
    if abs(gap)>5: cold.append("行%d 两遍 TTFT 差 %+.1f%%"%(row,gap))
    perf[row]=dict(ttft1=a[0],ttft2=b[0],tpot1=a[1],tpot2=b[1],d_ttft=dt,d_tpot=dp,cold_gap=gap)
    print("  行%-2d %-10s %10.1f %10.1f %+6.1f%% | %9.3f %9.3f %+6.1f%%"%(
        row,name,a[0],b[0],dt,a[1],b[1],dp))
if cold: print("\n  ⚠ 第 1、2 遍差异超过 5%%：%s（冷形状效应，以第 2 遍为准）"%("；".join(cold)))

print("\n【二】答得对不对 —— 简短问答\n")
t=post("/generate",{"text":"用一句话说明什么是快速排序。",
    "sampling_params":{"max_new_tokens":48,"temperature":0}})["text"]
body=t.split("</think>")[-1].strip()
zh_ok = len(set(body))>12 and "排序" in body
print("  中文生成: %s   %r"%("✅ 正常" if zh_ok else "❌ 异常", body[:50]))

GSM=os.environ.get("GSM8K_PATH","/workspace/datasets/gsm8k_test.jsonl")
if not os.path.exists(GSM):
    print("  [提示] 未找到 gsm8k 数据集，跳过正确性校验（可用 GSM8K_PATH 指定路径）")
    sys.exit(0)
rows=[json.loads(l) for l in open(GSM)]
def one(i,inc):
    r="Question: "+rows[i]["question"]+"\nAnswer:"
    return r+" "+rows[i]["answer"] if inc else r
PRE="".join(one(i,True)+"\n\n" for i in range(5))
def gold(a): return ast.literal_eval(a.split("####")[-1].strip().replace(",",""))
def ask(i):
    r=post("/generate",{"text":PRE+one(i,False)+" <think>\n\n</think>\n\n",
        "sampling_params":{"max_new_tokens":512,"temperature":0,"stop":["Question","Assistant:"]}})
    n=re.findall(r"\d+",r["text"].replace(",",""))
    return (ast.literal_eval(n[-1]) if n else None), r.get("meta_info",{}).get("finish_reason",{})
Q=list(range(5,15))
with ThreadPoolExecutor(10) as ex: got=list(ex.map(ask,Q))
ok=sum(1 for i,(p,_) in zip(Q,got) if p==gold(rows[i]["answer"]))
trunc=sum(1 for _,f in got if f.get("type")=="length" or f.get("matched")=="Question")
wrong=[i for i,(p,_) in zip(Q,got) if p!=gold(rows[i]["answer"])]
print("  gsm8k 10 题: %s 正确 %d/10  截断 %d%s"%(
    "✅" if ok>=9 else "⚠️" if ok>=8 else "❌", ok, trunc,
    ("  错题序号 %s"%wrong if wrong else "")))
if trunc: print("     ⚠ 有输出被截断，检查是否触发思考模式（见测试准则 4.1）")

print("\n"+"="*74)
mt=sum(perf[r]["d_ttft"] for r in perf)/3; mp=sum(perf[r]["d_tpot"] for r in perf)/3
v=lambda d:"明显变好" if d<-5 else "明显变差" if d>5 else "无显著变化（噪声内）"
print("  【有没有效】TTFT %+.1f%% → %s     TPOT %+.1f%% → %s"%(mt,v(mt),mp,v(mp)))
corr = zh_ok and ok>=9 and not fail
print("  【对不对】  %s"%("✅ 通过" if corr else "❌ 未通过 — "+"；".join(
    fail+([] if zh_ok else ["中文输出异常"])+([] if ok>=9 else ["gsm8k 仅 %d/10"%ok]))))
print("  用时 %.0f 秒"%(time.time()-t0)); print("="*74)
open(HIST,"a").write(json.dumps(dict(tag=TAG,ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
    perf=perf,zh_ok=zh_ok,gsm8k=ok,trunc=trunc,d_ttft=mt,d_tpot=mp,correct=corr),ensure_ascii=False)+"\n")
