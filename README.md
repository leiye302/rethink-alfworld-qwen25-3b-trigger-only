# ALFWorld / Qwen2.5-3B：只教触发，不教纠错

[CPU 检查记录](https://github.com/leiye302/rethink-alfworld-qwen25-3b-trigger-only/actions)

这是 RETHINK 的 **trigger-only** 消融仓库，不是完整方法的训练结果。基于同一冻结 3B 配方，唯一算法变化是移除 Bad 的纠正 action 和 EOS 监督；Good 原动作、独立 marker BCE 和后续 Actor 路由的 GRPO+OPSD 保持不变。

```text
Good: [H, R] masked → original sampled action + EOS
Bad : [H, R] masked → <RETHINK>       (no correction, no EOS target)
```

RETHINK 在 `</think>` 后、`<action>` 前触发，不是 action 后撤回。Bad 的训练输入在 marker 结束，纠正 action 不进入输入；原完整目标只留在离线数据中，用于保持同一批样本及原 CE 归一化系数。

对原完整后缀长度为 (n_i) 的 Bad 样本：

```math
L_i^{\mathrm{trigger}} = -\frac{1}{n_i}\log p_\theta(\texttt{<RETHINK>}\mid H_i,R_i)
    + 0.1\,\mathrm{BCE}\big(p_\theta(\texttt{<RETHINK>}\mid H_i,R_i),1\big).
```

不是把单个 marker CE 取平均后放大到权重 1。Good 保持完整原动作后缀的 sample-mean CE；H/R 没有直接 token CE，但共享参数仍会更新，并非冻结 reasoning。**不增加 SFT KL。**

## 固定设置

- 起点：`Qwen/Qwen2.5-3B-Instruct`，固定 revision 见 `configs/ablation.json`。
- SFT：10,000 Good + 6,000 Bad，每批 5:3，batch 8，LR 2e-6，1 epoch / 2,000 steps；沿用 4,000-step cosine horizon / 400-step warmup，而非在 2,000 步衰减至零。
- 二阶段：从本消融的 step2000 开始，150 updates，整轨迹 GRPO + SDAR OPSD，beta=5。只有 Actor 自己生成 marker 的位置额外向 Teacher 提供局部 hint。
- Teacher 是当前 Actor 的无梯度评分视角；固定 reference KL 模型不是 Teacher。Student 不看 hint，不替换采样动作。
- 评测：相同 ALFWorld **seen140** 清单/种子，最多50步、history 2、temperature 0.4、512-token 预算；没有外置 API、Analyzer 或 Teacher。

## 准备与运行

Linux/CUDA，当前入口配置为单张空闲 **80/96GB GPU**；CUDA/vLLM 实机显存与吞吐尚未验证。建议至少64GiB主机内存、180GiB空闲磁盘。不要在已有完整方法或另一消融的输出目录中运行。

完整安装、下载、CPU检查、断点恢复及最终评测见 [复现说明](docs/idea.md)。冻结 SFT 数据已包含在 `data/frozen_data.zip`；依赖从固定公开 SDAR commit 获取，不需要作者电脑、私有服务器或私人账户。模型和 ALFWorld 资源按说明自行下载。

```bash
# 完成 docs/idea.md 中的环境、模型、ALFWorld 和自己的 API 配置后：
bash scripts/reproduce.sh --check
bash scripts/reproduce.sh --run --allow-api-cost
# 或仅启动本消融的第一阶段：
bash scripts/train.sh sft
# SFT成功退出并核验step2000后：
bash scripts/train.sh opsd --allow-api-cost
```

`--allow-api-cost` 是二阶段提示调用的明确授权，不是费用上限。准备代码和 CPU 测试不发起付费调用。每100步记录 SFT dev loss，每500步保存权重并自由生成32条；step2000保存完整续训状态。二阶段每10次更新在固定seen140验证、每20次保存、保留最近两份完整断点和最终update150。

## 验证与比较边界

代码准备、CPU loss/路由隔离/打包测试与 GPU 训练是不同层次。**本仓库未训练新模型、未校准 API 质量、没有预填成绩。** 原始完整方法与本消融的 SFT 总 loss 目标不同，不能直接把两者总 loss 大小当作谁更好；应比较独立 route BCE、自由生成的协议/触发/纠正质量和固定任务成功率。

源码来源、逐项差异和未验证项见 [审核记录](docs/experiment_log.md)。只发布源码、公开冻结数据和配置，不发布私人日志、凭据或模型续训状态。新增代码沿用 [Apache-2.0](LICENSE)，上游 SDAR、Qwen、ALFWorld 保留各自许可。
