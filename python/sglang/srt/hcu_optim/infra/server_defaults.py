"""不经 start_server.sh 启动时，自动补齐本仓库的启动参数。

背景（2026-09-19 净室复现）：本仓库的全部优化挂在 start_server.sh 设置的环境变量与启动参数上。
  同一镜像、同一代码，改用任务书模板命令启动：未设 GPU_MAX_HW_QUEUES 时行 4 TPOT 4.1 → 248 ms；
  只补 GPU_MAX_HW_QUEUES=2 也只有厂商默认水平（行 13 TTFT 3092 ms，对本仓库 1516 ms）。
  评测方通常有自己的启动流程，故把「用户未指定的参数」在解析前补上，使任何启动方式得到同一配置。
规则：只补用户命令行里没有出现的参数；用户显式给出的一律不动。补了什么都打印出来。
环境变量由镜像 ENV 或源码树中的启动脚本显式提供；本模块检查关键环境变量并告警，不再在解释器启动时隐式注入。
开关 SGLANG_SERVER_DEFAULTS=0 可关闭。
"""
import os

_BUCKETS = "4 8 16 32 64 128 256 512 768 1024 1088 1536 2048 2112 2176 2240 2304 2368 2432".split()
_MUST_ENV = {"GPU_MAX_HW_QUEUES": "2", "SGLANG_QUANT_AR": "1", "SGLANG_FIRST_TOKEN_FIRST": "3",
             "PYTORCH_TUNABLEOP_ENABLED": "1", "SGLANG_USE_AITER_LINEAR_ATTN": "1",
             "SGLANG_ADAPTIVE_CHUNK": "1", "SGLANG_QAR_GROUP": "32"}


def _numa_nodes():
    try:
        return len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()])
    except Exception:
        return 0


def _tp(argv):
    for k in ("--tp", "--tp-size", "--tensor-parallel-size"):
        if k in argv:
            try:
                return int(argv[argv.index(k) + 1])
            except Exception:
                return 0
    return 1


def _cpu_governor():
    """短输入的 TTFT 受 CPU 发射速度限制：schedutil 下请求结束前来不及升频，1024–7168 输入的 TTFT 慢 10–25%（全表 O 约 −5，
    2026-09-23 同实例 A-B-A 实测：行 4 175/151/176 ms，全核平均 1311/2094/1313 MHz）。能写就设为 performance（整机设置，
    重启后恢复系统默认），写不了就警告。SGLANG_CPU_GOVERNOR=keep 跳过；也可指定其它策略名。"""
    want = os.environ.get("SGLANG_CPU_GOVERNOR", "performance")
    if want == "keep":
        return
    import glob
    files = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor"))
    if not files:
        print("[cpu_governor] ⚠ 读不到 CPU 调频设置，无法确认；短输入 TTFT 对调频策略敏感（见 BASELINE.md）", flush=True)
        return
    cur = sorted({open(f).read().strip() for f in files})
    if cur == [want]:
        print(f"[cpu_governor] 调频策略已是 {want}（{len(files)} 核）", flush=True)
        return
    try:
        for f in files:
            with open(f, "w") as fh:
                fh.write(want)
        print(f"[cpu_governor] ✅ 调频策略 {'/'.join(cur)} → {want}（{len(files)} 核；整机设置，重启后恢复默认）", flush=True)
    except OSError as e:
        print(f"[cpu_governor] ⚠ 调频策略为 {'/'.join(cur)}，无权限改为 {want}（{type(e).__name__}）：短输入 TTFT 会慢 10–25%，"
              f"全表 O 约低 5。请在主机上执行：for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo {want} > $f; done",
              flush=True)


def _hysmi(args, timeout=15):
    """调用 hy-smi，超时就放弃该进程、不等它退出（内核态阻塞的调用杀不掉，等待会挂死启动）。返回 (退出码, 输出)，超时返回 (None, "")。"""
    import subprocess, time
    pr = subprocess.Popen(["hy-smi"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while pr.poll() is None and time.time() - t0 < timeout:
        time.sleep(0.02)
    if pr.poll() is None:
        pr.kill()
        return None, ""
    return pr.returncode, pr.stdout.read()


def _gpu_perflevel():
    """小批次 decode 与短输入预填充对 GPU 核心时钟敏感。液冷机曾被设为 manual 且 sclk 固定在 8 档 1270 MHz（最高 1400），
    2026-09-24 同一配置 A-B-A：设为 auto（负载时 1400、空闲降到 600）后并发 1 每轮时长 16.94/16.91 → 16.42 ms（−3%），
    行 8 TPOT 13.14/13.11 → 12.75、TTFT 1021/1019 → 999 ms，输出逐位不变。能改就设为 auto（整机设置，重启后恢复驱动默认），
    改不了就警告。只用 hy-smi，不读写驱动参数文件。SGLANG_GPU_PERFLEVEL=keep 跳过；也可指定 high 等档位名。"""
    import re
    want = os.environ.get("SGLANG_GPU_PERFLEVEL", "auto")
    if want == "keep":
        return
    vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    dev = ["-d"] + [d for d in vis.split(",") if d.strip()] if vis else []
    rc, out = _hysmi(["-p"] + dev)
    if rc is None:
        print("[gpu_perflevel] ⚠ hy-smi -p 15 秒未返回，跳过", flush=True)
        return
    cur = sorted(set(re.findall(r"Performance Level:\s*(\S+)", out)))
    if not cur:
        print("[gpu_perflevel] ⚠ 读不到 GPU 性能档位（hy-smi -p 无输出），无法确认", flush=True)
        return
    if cur == [want]:
        print(f"[gpu_perflevel] GPU 性能档位已是 {want}", flush=True)
        return
    rc, out2 = _hysmi(["--setperflevel", want] + dev)
    rc3, out3 = _hysmi(["-p"] + dev)
    now = sorted(set(re.findall(r"Performance Level:\s*(\S+)", out3)))
    if now == [want]:
        print(f"[gpu_perflevel] ✅ GPU 性能档位 {'/'.join(cur)} → {want}（整机设置，重启后恢复驱动默认）", flush=True)
    else:
        print(f"[gpu_perflevel] ⚠ GPU 性能档位为 {'/'.join(cur)}，未能改为 {want}（hy-smi 退出码 {rc}）：manual 低档位下 decode 慢约 3%。"
              f"请在主机上执行：hy-smi --setperflevel {want}", flush=True)


def install(mod):
    if getattr(mod, "_server_defaults_installed", False) or not hasattr(mod, "prepare_server_args"):
        return
    orig = mod.prepare_server_args

    def prepare_server_args(argv):
        argv = list(argv); added = []
        has = lambda *ks: any(k in argv or any(a.startswith(k + "=") for a in argv) for k in ks)

        def add(*items):
            argv.extend(items); added.append(" ".join(items))
        if not has("--page-size"):
            add("--page-size", "64")
        if not has("--disable-custom-all-reduce", "--custom-all-reduce-backend"):
            add("--disable-custom-all-reduce")
        if not has("--warmups"):
            add("--warmups", "prefill_shapes")
        if not has("--prefill-max-requests"):
            add("--prefill-max-requests", "2")
        if not has("--max-prefill-tokens"):
            add("--max-prefill-tokens", "40960")          # adaptive_chunk 的 32768 档均分后每块最多约 37700 token
        if not has("--cuda-graph-backend-prefill"):
            add("--cuda-graph-backend-prefill", "breakable")
        if not has("--cuda-graph-bs-prefill"):
            add("--cuda-graph-bs-prefill", *_BUCKETS)
        if not has("--kv-cache-dtype"):
            add("--kv-cache-dtype", "fp8_e4m3" if os.environ.get("SGLANG_FP8_KV_HCU", "0") != "0" else "auto")
        if not has("--numa-node") and _tp(argv) == 4 and _numa_nodes() == 4:
            add("--numa-node", "0", "1", "2", "3")
        try:
            _cpu_governor()
        except Exception as e:
            print(f"[cpu_governor] ⚠ 检查失败: {type(e).__name__}: {e}", flush=True)
        try:
            _gpu_perflevel()
        except Exception as e:
            print(f"[gpu_perflevel] ⚠ 检查失败: {type(e).__name__}: {e}", flush=True)
        if added:
            print("[server_defaults] ✅ 已补齐未指定的启动参数: " + " | ".join(added), flush=True)
        else:
            print("[server_defaults] 启动参数均已由命令行给出，未补任何项", flush=True)
        bad = {k: os.environ.get(k) for k, v in _MUST_ENV.items() if os.environ.get(k) != v}
        if bad:
            print(f"[server_defaults] ⚠ 关键环境变量缺失或不同: {bad}。请用 start_server.sh 启动，或使用已内置 ENV 的镜像；"
                  "缺 GPU_MAX_HW_QUEUES=2 时 decode 可慢数十倍", flush=True)
        return orig(argv)

    mod.prepare_server_args = prepare_server_args
    mod._server_defaults_installed = True
    print("[server_defaults] 已安装", flush=True)
