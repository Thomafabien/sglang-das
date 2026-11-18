# Qwen3.6 K100_AI 优化源码合入说明

## 1. 交付目标与架构

本分支以 `HYGON-AI/sglang-das@v0.5.15.post1_dev` 为客户基线，把原交付目录中的优化代码、原生内核、调优表和验证脚本纳入 SGLang 源码树。最终运行不依赖 `sitecustomize.py`、meta-path loader 或外置 patch 目录。

所有一级功能开关均为**显式启用、默认关闭**：

- 未设置、空值、`0`、`false`、`off`、`no`、`none` 或 `null`：不导入优化模块，目标模块保持原始实现；
- 设置为其他值（通常为 `1`）：目标 SGLang 模块定义完原实现后，显式注册对应优化；
- 实现加载失败时记录完整异常，便于单项定位；
- 原实现、优化实现、原生库与配置由同一个 Git 分支和 Python 包交付。

代码结构：

- `python/sglang/srt/hcu_optim/integration.py`：目标模块、实现模块和一级开关的唯一注册表；
- 17 个 SGLang 目标模块末尾的 `_apply_hcu_optim(sys.modules[__name__])`：显式源码接入点；
- `python/sglang/srt/hcu_optim/{ops,kv,sched,comm,infra}`：按领域拆分的实现；
- `python/sglang/srt/hcu_optim/lib`：随包交付的 K100_AI 原生库；
- `python/sglang/srt/hcu_optim/native`：原生扩展源码与构建脚本；
- `k100opt/config`：MoE/TunableOp 调优表；
- `k100opt/bench`：性能与正确性验证脚本。

旧方案由 Python 启动过程全局拦截 import；新方案由 SGLang 源码自己声明和调用扩展点，`PYTHONPATH` 顺序不再暗中替换模块。

## 2. 合入模块总表

| 优化模块 | 一级开关 | SGLang 接入位置 | 修改内容与保护条件 | 预期影响 |
|---|---|---|---|---|
| Conv1D 配置 | `SGLANG_CONV1D_CFG` | causal-conv1d 源码直接分支 | K100_AI 形状选择；关闭用原配置 | 降低 GDN Conv1D 开销 |
| GDN chunk-h | `SGLANG_CHUNK_H_CFG` | `fla/chunk_delta_h.py` 直接分支 | 覆盖 BV/warps/stages | 改善状态更新 kernel |
| GDN no-zero | `SGLANG_GDN_NOZERO` | `fla/chunk_o.py` 直接分支 | 跳过可证明无用的清零 | 减少 kernel 和显存写 |
| GDN kernel 配置 | `SGLANG_GDN_CFG` | GDN Triton 源码直接分支 | K100_AI kernel 参数 | 改善线性注意力 |
| LightOp 兼容 | `SGLANG_LIGHTOP_COMPAT` | `moe_align_block_size.py` | 兼容新 LightOp 符号；旧符号存在则不改 | 避免 API 迁移导致启动失败 |
| MoE 快速对齐 | `SGLANG_FAST_MOE_ALIGN` | `fused_moe.py` | 仅 EP=1、无 EPLB/DeepEP 时跳过哨兵映射 | 减少对齐辅助 kernel |
| MoE 激活展开 | `SGLANG_ACT_FLAT` | `fused_moe.py` | 大 M、小 D 使用扁平 Triton SiLU×Mul；其他形状回退 | 改善窄行 wave 利用率 |
| MoE w13 预打包 | `SGLANG_MOE_PACK_W13` | `fused_moe.py`、`unquant.py` | 按 32×128 tile 预排布；不支持组合明确拒绝 | 改善无地址散列卡跳行读取 |
| Dense 预打包 | `SGLANG_DENSE_PACK` | `unquant.py` | 线性权重使用 K100_AI 友好布局 | 提高连续访存效率 |
| M3 融合 | `SGLANG_M3_FUSE` | `models/qwen2_moe.py` | 合并 Qwen2-MoE 指定子路径 | 减少 launch/中间结果 |
| FP8 KV HCU | `SGLANG_FP8_KV_HCU` | `overrides.py`、`flashattention_interface.py`、`server_args.py` | HCU FP8 KV、paged-attention 与格式选择 | 降低 KV 容量/带宽 |
| Fused draft top-k | `SGLANG_FUSED_DRAFT_TOPK` | `eagle_worker_v2.py` | 草稿阶段 top-k 融合 | 减少 speculative 小算子 |
| Draft vocab parallel | `SGLANG_VPA_DRAFT` | `eagle_worker_v2.py`、`logits_processor.py`、`qwen3_5_mtp.py` | 草稿 vocab 并行 | 减少全量 logits 聚合 |
| Verify vocab parallel | `SGLANG_VPA_VERIFY` | `eagle_worker_v2.py`、`logits_processor.py` | 验证阶段 vocab 并行 | 降低 verify 通信与显存 |
| Warm batch | `SGLANG_WARM_BATCH` | `entrypoints/warmup.py` | 按生产形状分组预热 | 降低首轮编译 TTFT 抖动 |
| Adaptive PMR | `SGLANG_ADAPTIVE_PMR` | `schedule_policy.py` | 按输入长短动态选择 PMR | 改善混合负载调度 |
| Long admit | `SGLANG_LONG_ADMIT` | `schedule_policy.py` | 长请求准入比例/阈值 | 保护短请求尾延迟 |
| First-token-first | `SGLANG_FIRST_TOKEN_FIRST` | `scheduler.py` | 优先尚未产出首 token 的请求 | 改善 TTFT |
| Adaptive chunk | `SGLANG_ADAPTIVE_CHUNK` | `scheduler.py` | 按负载选择 prefill chunk 档位 | 平衡 TTFT、吞吐和显存 |
| Pair prefill | `SGLANG_PAIR_PREFILL`（组合开关） | first-token-first/quant-AR 内部调用 | 相邻 prefill chunk 计算/通信交错；依赖 greenlet | 隐藏通信开销 |
| Tokenizer pool | `SGLANG_TOK_POOL` | `tokenizer_manager.py` | 长文本分段/线程池 tokenization | 降低长输入 CPU 阻塞 |
| Fused KV write | `SGLANG_KV_WRITE_FUSED` | `memory_pool.py` | 融合 KV cache 写入 | 减少写缓存 kernel |
| MTP RoPE Triton | `SGLANG_MTP_ROPE_TRITON` | `mrope.py` | MTP 走专用 Triton RoPE | 减少小算子和临时张量 |
| Quantized all-reduce | `SGLANG_QUANT_AR` | `parallel_state.py` | 指定规模启用量化通信 | 降低 TP 通信量 |
| Server defaults | `SGLANG_SERVER_DEFAULTS` | `server_args.py` | K100_AI 推荐默认值与运行检查 | 避免漏设环境造成漂移 |
| Fast import | `SGLANG_FAST_IMPORT` | `sglang/__init__.py` | 显式优化外部 `vllm_hcu.patch_utils` 导入 | 缩短 worker import 时间 |

`safe_align.py`、`pair_prefill.py`、`pa_py.py` 是组合功能的内部实现，分别由 pair-prefill 和 FP8-KV 按需加载，不单独改变原路径。

## 3. 二级调优与诊断变量

一级开关决定是否改变原路径；以下二级变量只在对应一级开关已开启后生效。

### MoE 与 GDN

- `SGLANG_ACT_FLAT_MIN`（默认 `20000`）、`SGLANG_ACT_FLAT_BLOCK/WARPS/MAX_D/AUDIT`；
- `SGLANG_MOE_CONFIG_DIR`、`SGLANG_M3_FUSE_EVERY`；
- `SGLANG_CHUNK_H_BV/WARPS/STAGES`；
- `SGLANG_GDN_CHUNK_H_BV/NUM_WARPS/NUM_STAGES`（兼容旧变量名）；
- `SGLANG_GDN_NOZERO_AUDIT`、`SGLANG_USE_AITER_LINEAR_ATTN`。

### FP8 KV / Paged Attention

- `SGLANG_PA_OWN`、`SGLANG_PP_OWN`：自带 paged/prefill 路径；
- `SGLANG_PA_FMT`、`SGLANG_PA_EXT`：格式和原生扩展；
- `SGLANG_PA_CHECK/TRACE/DIAG_BT`：正确性与调用诊断；
- `SGLANG_PA_G`、`SGLANG_PA_NSPLIT`：原生内核分组/切分；
- `SGLANG_HCU_KERNEL_LIB_DIR`：随包 `.so` 目录覆盖。

### 调度

- `SGLANG_ADAPTIVE_CHUNK_TIERS/EVEN`；
- `SGLANG_ADAPTIVE_PMR_THRESH/LONG/SHORT/FIXCHUNK`；
- `SGLANG_LONG_ADMIT_THRESH/K/POLL`；
- `SGLANG_PAIR_PREFILL_MIN_TOKENS/STREAMS/STRICT/SAFE_ALIGN/MOE_OVERLAP`；
- `SGLANG_PAIR_PREFILL_ALLOW_LOGPROB/DEBUG/TIMING/EXCL/NO_COMM_ORDER`。

### 通信与基础设施

- `SGLANG_QAR_GROUP/MIN_NUMEL/IN_GRAPH/CFG_FILE/OFF_FILE/ERR_LOG`；
- `SGLANG_TOK_POOL_MIN_CHARS/SEGS/LOG_N`；
- `SGLANG_WARM_BATCH_GROUPS`；
- `SGLANG_CPU_GOVERNOR`、`SGLANG_GPU_PERFLEVEL`。

## 4. 启动与 A/B

原始路径：新 shell 中不设置上述一级开关，只设置：

```bash
export PYTHONPATH=/workspace/sglang-das-k100opt/python
python3 -u -m sglang.launch_server ...
```

单项测试只设置一个一级开关，例如：

```bash
SGLANG_ACT_FLAT=1 SGLANG_ACT_FLAT_MIN=20000 \
  python3 -u -m sglang.launch_server ...
```

组合脚本：

```bash
bash k100opt/start_gdn_moe_server.sh       # 已验证的 Conv1D+GDN+MoE
bash k100opt/start_all_optimizations.sh     # 全模块集成候选
```

全模块脚本用于集成验证，不应替代逐项 A/B。FP8 KV、pair-prefill、量化通信和调度策略会改变运行条件，必须分别验证正确性后再纳入最终性能配置。

## 5. 验证顺序

```bash
python3 k100opt/verify_source_integration.py
python3 -m py_compile <全部变更 Python 文件>
git diff --check
```

容器中依次执行：

1. 本地与容器 `git rev-parse HEAD` 一致；
2. 本地与容器 `git rev-parse 'HEAD^{tree}'` 一致；
3. 所有一级开关未设置时导入目标模块，确认无优化生效日志；
4. 单项开启，检查对应“已安装/首次真正生效”日志；
5. 服务健康检查；
6. smoke 跑两次，第一遍丢弃、第二遍记录；
7. 与 clean、full-opt 两个固定端点比较 TTFT、TPOT、输出吞吐和正确性。

## 6. 合入设计的影响

- **可审查**：每个目标模块有明确接入点，注册表能回答“谁修改了谁”；
- **可回滚**：一级开关默认关闭，关闭时不 import 对应优化；
- **可打包**：实现、原生库和配置随仓库交付，不依赖容器隐藏文件；
- **可定位**：功能开关与诊断参数分离，可逐项测试；
- **避免污染**：无 `sitecustomize` 和全局 import hook；
- **保留原路径**：目标模块先定义完整原实现，再按显式开关注册优化。

最终可公开性能数据必须来自同节点、同卡、同模型、同服务参数、两遍 smoke 的第二遍记录。本文件中的收益描述是优化机制和已有验证方向，不替代最终基准表。
