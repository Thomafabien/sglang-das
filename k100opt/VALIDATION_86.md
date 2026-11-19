# 节点 86 全源码集成验证记录

## 验证对象

- 节点：`kme86`，容器：`qwen36-mjg`
- 分支：`feat/qwen36-k100ai-opt`
- 被测提交：`33e7e7a4227e01617593b8091035e08f870a28a0`
- 被测 tree：`43251a916928d02a07570d6c7236db2da0474556`
- 模型：`/model/Qwen3.6-35B-A3B`
- GPU：`4,5,6,7`，TP=4，端口 `30010`
- 启动配置由源码仓库外的独立部署脚本提供。

本地与容器在启动前同时核对 commit 和 tree，两个值均一致。容器通过 `git pull --ff-only` 同步，没有覆盖本地修改。

## 源码与 clean-path 检查

```text
PASS: 17 target modules; 26 feature bindings; default-off and legacy-overlay checks passed
DEFAULT_OFF_TARGET_IMPORTS_OK 17
OPT_IMPLEMENTATIONS_LOADED []
```

这证明：旧 `sitecustomize`/kpatch overlay 已删除；所有一级开关未设置时，17 个目标模块可正常导入，且没有加载任何优化实现模块。

## 服务启动检查

服务通过 `/v1/models` 健康检查。日志确认以下路径首次真实生效：

- Conv1D 配置、chunk-h；
- quantized all-reduce；
- MoE fast-align、w13 pack、M3、act-flat；
- fused KV write、FP8 KV prefill/decode；
- dense pack、MTP RoPE；
- draft/verify vocab parallel；
- warm batch；
- first-token-first 模式 3。

观察项：`gdn_nozero` 的保护逻辑发现当前 `chunk_o` 已不调用预期的 `zeros_like`，因此本次没有实际替换，并输出告警；服务按当前原实现继续运行。该项不应计入本轮实测收益，后续需按现版 `chunk_o` 重新确认是否仍有可删除清零。

## 两遍 smoke

第一遍只作热身，第二遍是正式记录。每个 smoke 内部的三个测点仍各跑两次并以内部第二次计数。

### 第一遍（丢弃）

| 场景 | TTFT-2 ms | TPOT-2 ms | 输出吞吐 tok/s |
|---|---:|---:|---:|
| 行4 短输入·单并发 | 138.1 | 3.890 | 248.07 |
| 行13 中输入·高并发 | 1491.9 | 13.260 | 975.07 |
| 行29 长输入·单并发 | 1186.7 | 4.060 | 191.48 |

### 第二遍（正式）

| 场景 | TTFT-2 ms | vs clean | vs full-opt | TPOT-2 ms | vs clean | vs full-opt | 输出吞吐 tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 行4 短输入·单并发 | 140.5 | -34.3% | +0.9% | 3.900 | -33.8% | -4.9% | 247.00 |
| 行13 中输入·高并发 | 1489.1 | -55.4% | -0.8% | 13.260 | -36.7% | -2.5% | 974.52 |
| 行29 长输入·单并发 | 1198.4 | -45.2% | -0.1% | 4.110 | -32.1% | -3.7% | 189.29 |

平均结果：

- vs clean：TTFT `-45.0%`，TPOT `-34.2%`，均明显变好；
- vs full-opt：TTFT `-0.0%`，TPOT `-3.7%`，处于 5% 噪声带内；
- 中文生成正常；
- GSM8K `9/10`，截断 `0`；
- smoke 正确性结论：通过。

## 日志位置

日志保存在容器外持久目录：

```text
/workspace/k100opt/results/source-integration-33e7e7a4/service.log
/workspace/k100opt/results/source-integration-33e7e7a4/smoke-first-discard.log
/workspace/k100opt/results/source-integration-33e7e7a4/smoke-second-record.log
```

早一版 `f908a0c1` 的日志保留在：

```text
/workspace/k100opt/results/source-integration-f908a0c1/
```

最终服务未停止，仍在容器内监听 `30010`，便于继续进行单模块 A/B。
