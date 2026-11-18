# Qwen3.6 K100_AI source-integrated optimizations

本目录配合 `python/sglang/srt/hcu_optim` 提供 K100_AI 优化的配置、启动、验证和复现材料。优化已进入 SGLang 源码包，不再依赖 `sitecustomize.py` 或外置 `PYTHONPATH` patch。

- 完整合入说明：`SOURCE_INTEGRATION.md`
- 手工复现步骤：`MANUAL_REPRO.md`
- 源码集成自检：`verify_source_integration.py`
- Conv1D+GDN+MoE：`start_gdn_moe_server.sh`
- 全模块候选组合：`start_all_optimizations.sh`
- 性能/正确性脚本：`bench/`
- MoE 与 TunableOp 配置：`config/`

所有一级优化开关默认关闭。直接运行原 SGLang 启动命令得到 clean 路径；使用上述启动脚本才会显式开启相应优化。
