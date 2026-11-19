# Qwen3.6 K100_AI source-integrated optimizations

本目录配合 `python/sglang/srt/hcu_optim` 提供 K100_AI 优化的配置、源码自检和验证材料。优化已进入 SGLang 源码包，不再依赖 `sitecustomize.py` 或外置 `PYTHONPATH` patch。

- 完整合入说明：`SOURCE_INTEGRATION.md`
- 节点 86 验证结果：`VALIDATION_86.md`
- 源码集成自检：`verify_source_integration.py`
- 性能/正确性脚本：`bench/`
- MoE 与 TunableOp 配置：`config/`

所有一级优化开关默认关闭。服务启动命令、节点路径、模型路径、GPU 和端口配置属于部署材料，必须独立维护，不放入源码仓库或官网源码包。
