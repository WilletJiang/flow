# 条件流初始化

目标：对每个时间步 t 生成后验样本 {x_t^(k)} 及其经验均值/协方差 (μ_t, P_t)，作为下游 EM / RTS / PF / VI 的初始化。

核心设计
- 上下文编码器：将 token = concat[y_t, feat(G_t), feat(R)] 送入 Transformer 编码器（正弦位置编码），得到上下文 c_t。
- 条件流：Conditional RealNVP（交替掩码的仿射耦合，带上下文），采样/评估均为并行路径；尾部仿射头支持岭回归热身。
- 损失：观测负对数似然（线性高斯）、先验 KL/L2、弱平滑正则；可选熵项（支持子采样）。
- 推理：大规模采样 + 向量化二阶矩估计 μ,P，PSD 修复；可选温度校准。

目录结构
- `src/flow_init/encoder.py`：上下文编码器（TransformerEncoder + SDPA 布尔掩码）。
- `src/flow_init/flows.py`：条件 RealNVP（仿射耦合 + 上下文）。
- `src/flow_init/losses.py`：观测 NLL、先验 KL、平滑项。
- `src/flow_init/train.py`：热身与训练（支持 autocast/compile、熵子采样）。
- `src/flow_init/infer.py`：采样、向量化 μ/P、温度校准。
- `src/flow_init/utils.py`：Cholesky/Welford/岭回归等数值工具。

安装
```bash
pip install -r requirements.txt
# 说明：
# - CUDA 机器建议安装 xFormers（已在 requirements.txt 中）。
# - macOS M2 (MPS) 上 xFormers 可能无法安装，不影响运行（自动回退到 PyTorch SDPA）。
```

快速上手
```shell
python scripts/run_init.py --epochs_warmup 8 --epochs 40 --K 512 --device cpu --full_context 1 --window -1
```

配置与性能开关（强烈建议 CUDA 上启用）
- 配置定义：`src/flow_init/config.py`

1) `Config.perf.autocast_bf16: bool`（默认 True）
- 含义：CUDA 设备上以 bfloat16 自动混合精度包裹编码器与流（推/训），显著提升吞吐与显存效率。

2) `Config.perf.compile: bool`（默认 True）
- 含义：尝试对 Encoder/Flow 使用 `torch.compile(mode='max-autotune', dynamic=True)`；不可用时自动回退。

3) `TrainConfig.entropy_subsample: float ∈ [0,1]`（默认 1.0）
- 含义：训练期熵项/`log_prob` 的子采样比例。将其设为 0.25~0.5 可明显降低计算开销，通常对收敛影响极小。

4) `EncoderConfig.use_xformers: bool`（默认 False）
- 含义：预留开关，用于将注意力替换为 xFormers 的 memory‑efficient 内核。当前默认仍使用 PyTorch SDPA（不开启不改变行为）。

注意力与内核说明
- 我们使用“布尔掩码”的注意力（True 表示屏蔽），便于命中 PyTorch 2.x 的 SDPA 高效核（Flash/Memory‑Efficient/cudnn 自动选择）。
- 包初始化时会尝试启用更高效的 SDPA 后端；若不支持会自动回退，无需手工干预。

设备建议
- CPU（开发/小规模验证）：无 CUDA/Flash 内核，MPS 下默认 fp32 即可（自动回退），无需 xFormers。
- GPU（本地训练）：建议保持 `autocast_bf16=True`、`compile=True`，并将 `entropy_subsample=0.25` 作为强基线。
- H800及以上（云端大型训练）：Hopper 上 Flash SDPA 性能最佳；在显存允许范围内增大 batch/K 可进一步提升吞吐。

xFormers vs TransformerEncoder + SDPA 的区别
- 共同点：都能实现精确注意力，并利用 GPU 高效内核；在全局/因果注意力上，PyTorch SDPA 与 xFormers 的吞吐通常相近。
- PyTorch TransformerEncoder + SDPA：
  - 内置、零额外依赖；与 `scaled_dot_product_attention` 深度集成，自动选择 Flash/Memory‑Efficient/cudnn 等后端；维护与兼容性最好。
  - 适合标准全局/因果/窗口注意力（配合布尔掩码）。
- xFormers：
  - 提供 `memory_efficient_attention` 等多种内核、支持更多稀疏/块稀疏/窗口化模式（如 Swin/轴向/块稀疏）。
  - 在复杂稀疏模式与大序列上显存占用更低、前后向更稳；但需要额外安装，部分平台（如 M2）不可用。
> 建议：标准 Encoder 首选 PyTorch SDPA；若需要特定稀疏/窗口化模式，再开启 `use_xformers=True`。

评估与可视化
- 覆盖曲线/PIT/时间覆盖：`src/flow_init/eval.py:1`
- 推理输出：`infer_posterior` 返回 `{"mu", "P", "samples"}`，温度校准可用 `calibrate_temperature`。

重要代码位置
- 编码器（布尔掩码 + SDPA）：`src/flow_init/encoder.py:52`
- 条件 RealNVP：`src/flow_init/flows.py:1`
- 训练（autocast/compile/熵子采样）：`src/flow_init/train.py:1`
- 推理（向量化二阶矩）：`src/flow_init/infer.py:14`
- 数值工具/Cholesky/岭回归：`src/flow_init/utils.py:1`

兼容性与数值
- 已移除 float64 强制，默认 dtype 随环境/Autocast；Cholesky 等路径在 fp32 下配合 `eps*I` 修正稳定。
- 若对极端数值稳定性有更高要求，可手动切回 fp32 训练与评估（关闭 autocast）。

