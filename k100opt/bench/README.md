# 压测套件与测量口径

`BASELINE.md` 里那组分数（O 122.74 / P 149.59）就是用本目录的脚本测出来的。
**如果测出的数字与基线差很多，先逐条核对下面的「六条口径」——
经验上数字对不上，绝大多数不是服务端配置问题，而是压测侧口径不同。**

---


## 复测分四级（由快到慢，建议按顺序做）

| 命令 | 内容 | 用时 | 用途 |
|---|---|---|---|
| `bash bench/run_all.sh smoke` | 3 个测点 | 约 8 分钟 | 确认服务正常、量级对 |
| `bash bench/run_all.sh quick` | 10 个代表测点 × 1 遍（覆盖 1024–131072 各输入档与并发 1/2/4/16） | 约 12 分钟 | 逐行与基准对照，看哪一档偏了 |
| `bash bench/run_all.sh once` | 45 测点 × 1 遍 | 约 35 分钟 | O/P 的单遍值。R16 时期实测单遍与三遍相比 O 低约 1 点、P 相差不到 0.3；R18 修复首遍效应后未复核 |
| `bash bench/run_all.sh full` | 45 测点 × 3 遍 | 约 110 分钟 | 正式成绩 |

每一级都会：自检环境 → 启动服务 → 打印 `[switches]` 与补丁装载记录（液冷应为 22 项；风冷档位 25 项）→ 测量 → 打印**与本仓库基准值（`bench/baseline.json`，液冷 R21 单遍）的逐行对照**，
偏慢的测点会被标出并给出排查顺序。抽测的 O/P 只是所测测点的均值，同时给出同一批测点的基准均值以便比较，不等于全表成绩。
精度评测见 `bench/eval/README.md`。

## 快速开始

### 推荐：一条命令

```bash
bash <repo>/bench/run_all.sh smoke    # 3 测点快速检查，约 8 分钟（含启动）
bash <repo>/bench/run_all.sh full     # 45 测点 × 3 遍正式测量，约 110 分钟
```

`run_all.sh` 把下面「六条口径」里能自动化的部分都做了：

- 开测前检查残留进程与各卡 HCU 利用率，非独占会告警
- 写 PID 锁，**已有实例在跑时拒绝启动**（避免两个压测并跑）
- 启动服务并等待就绪，启动失败直接报错退出
- 打印生效性自证：`[switches]` 行、22 条补丁装载记录、MoE 配置的实际来源路径
- 用正确的 `PYTHONPATH` 与热口径执行测量，随后评分
- 结束时停掉服务

### 或者分步执行

```bash
# 1) 启动服务（另开终端，等出现 "fired up and ready"）
bash <repo>/start_server.sh

# 2) 全量测量
cd <repo>/bench
PYTHONPATH=/workspace/sglang-das/python python3 run45.py

# 3) 评分
PYTHONPATH=/workspace/sglang-das/python python3 score45.py
```

可用环境变量：`MODEL_PATH`、`SGLANG_PYTHON`、`HOST`、`PORT`、`OUTDIR`（默认 `./out`）、
`PASSES`（默认 3，评分口径要求，不要改）。

---

## 六条口径（数字对不上先查这里）

### 一、`PYTHONPATH` 必须指向 sglang-das（最常见的原因）

```bash
PYTHONPATH=/workspace/sglang-das/python python3 run45.py
```

不带这个，`import sglang` 会落到镜像里 pip 安装的 sglang 0.5.12。
**两个版本的随机种子默认值不同**，实测对比：

| | 压测入口 | `--seed` 默认值 |
|---|---|---|
| pip 装的 sglang 0.5.12 | `sglang.bench_serving`（真实现） | **1** |
| | `sglang.benchmark.serving` | 该模块不存在 |
| sglang-das | `sglang.benchmark.serving`（真实现） | **42** |
| | `sglang.bench_serving` | 转发壳，转到上面那个，同为 42 |

种子不同意味着**随机生成的 prompt 完全是另一批**，token 分布不同，
测出的 TTFT/TPOT 与基线、与 45 行表格里的 L20 参考值都**不可比**。

**这是个静默陷阱**：用 `python3 -m sglang.bench_serving` 且不带 PYTHONPATH 时，
程序不会报错，会正常跑完并给出看似合理的数字。

`run45.py` 启动时会自检并打印 `[自检] sglang 来源: ...`。
若落到 site-packages 会直接退出（退出码 2）并提示。
**请确认这一行指向 sglang-das；这是数字能否对齐的前提。**

### 二、不要传 `--warmup-requests`

本项目 45 测点的正式口径是**热口径**：不传该参数，使用其默认值 1。
这是为了对齐实验室的测试命令。

若显式传 `--warmup-requests 0`（冷口径），TTFT 约高 5%，O 会明显偏低。
`run45.py` 已按热口径固定，不要自行加这个参数。

> 注：本目录的 `smoke.py` 用的是冷口径（传 0），它只用于改动前后的相对比较，
> **其绝对值不能与 45 行表格或 BASELINE.md 的分数对照。**

### 三、3 遍全平均，不丢首遍

评分口径是三遍全部取平均。丢掉首遍会使 O 偏高约 1.3 点。
`PASSES` 默认 3，请勿改动。

### 四、机器必须独占，且只能有一个压测实例

```bash
# 开测前确认没有别的服务或压测在跑
ps -eo pid,etime,cmd | grep -E "launch_server|benchmark.serving|run45" | grep -v grep
hy-smi | grep -E "^[0-3] "      # HCU% 应为 0
```

两个压测进程同时打同一个服务会造成**随测点推进而累积的劣化**：
本项目曾误启动两个实例，前几行正常，到后段测点 TTFT 劣化达 2.5 倍。
这种劣化的特征是"越往后越差"，很容易被误判成某些测点本身慢。

### 五、确认服务端补丁真的装上了

启动日志里应有：

```
[switches] ... SGLANG_FAST_IMPORT=1 ... NCCL_MAX_NCHANNELS=16 NCCL_MIN_NCHANNELS=16 ...
```

以及 22 条装载记录：

```bash
grep -oE "\[[a-z_0-9]+\] 已安装" <服务日志> | sort | uniq -c
#   期望 22 项：act_flat / adaptive_chunk / adaptive_pmr / chunk_h_cfg / fast_import / fast_moe_align /
#              first_token_first / fp8_kv_hcu / fused_draft_topk / gdn_cfg / gdn_nozero / lightop_compat /
#              long_admit / m3_fuse / mtp_rope_triton / pair_prefill / quant_ar / server_defaults / tok_pool /
#              vocab_par_draft / vocab_par_verify / warm_batch
```

补丁在条件不满足时会静默回退到原实现，**只有装载记录才能确认它真的生效**。
其中缺少 `fast_import` 会使并发 16 的测点劣化约 10%。

另可核对服务加载的 MoE 配置来自本仓库而非 sglang 源码树：

```bash
grep -o "Using MoE kernel config from .*" <服务日志> | sort -u
#   应指向 <repo>/config/moe_config/...，共两条（主配置与 _down）
```

### 六、其余必须一致的压测参数

`run45.py` 里已固定，此处列出以便核对：

| 参数 | 值 |
|---|---|
| `--dataset-name` | `random-ids` |
| `--random-range-ratio` | `1`（定长，不做长度抖动） |
| `--request-rate` | `inf`（一次性全部发出） |
| `--num-prompts` | 等于 `--max-concurrency` |
| `--tokenizer` | 与服务端同一模型路径 |

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `run_all.sh` | 一条命令跑完：自检 → 启动 → 自证 → 测量 → 评分 → 停服务 |
| `spec45.json` | 45 个测点的 (输入长度, 输出长度, 并发度) 与 L20 参考值 |
| `run45.py` | 全量测量，45 测点 × 3 遍，输出 `out/formal_results.json` |
| `score45.py` | 评分，输出 `out/scored45.json` 与逐行表、分组汇总 |
| `smoke.py` | 3 测点快速检查（约 2 分钟），用于改动前后的相对比较 |

## 预期结果

用本仓库的 `start_server.sh` 启动、按上述口径测量，预期：

```
O = 122.7 ± 0.5      P = 149.6（R18 起冷启动即稳态，无首遍惩罚）
三遍离散度(TTFT) 中位约 0.5%
```

误差范围是三遍离散度与跨实例漂移之和。
**若偏差明显超出此范围，请先按「六条口径」自查，并把 `[自检] sglang 来源` 那一行
与 22 条装载记录一起发出来，便于定位。**
