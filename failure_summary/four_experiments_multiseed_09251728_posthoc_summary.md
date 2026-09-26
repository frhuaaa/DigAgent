# DiagAgent 三 Seed 四组实验事后总结（09251728）

> 整理日期：2026-09-25。本文件是研究者事后分析，包含隔离测试集结果。
> 其中任何测试指标都不得进入 CLEM、RASS、FAMA、RAPA、Validation Gate、
> accepted-state、停止逻辑或 Memory Bank，也不得用于追溯修改既有轨迹。

## 1. 实验范围与口径

本次总结覆盖以下四条正式轨迹：

```text
runs/csi1000_deepseek_c2c/2024/lstm/09251728
runs/csi1000_deepseek_o2o/2024/lstm/09251728
runs/csi500_deepseek_c2c/2024/lstm/09251728
runs/csi500_deepseek_o2o/2024/lstm/09251728
```

四条轨迹均使用 ensemble seeds `[0, 1, 2]`。每个 seed 独立按
`train_valid IC` 选择 checkpoint，然后对三个预测做每日截面 z-score 后等权平均。
四条轨迹均完成 10 个 adaptive rounds，并因
`ADAPTIVE_ROUND_BUDGET_EXHAUSTED` 正常停止。

本文把“成功”定义为：最终 accepted 配置相对 Round 0 不仅提高
`agent_valid` 绝对 Sharpe，而且在隔离测试期保持同方向改善。仅流程正常结束不等于研究成功。

## 2. 总结表

| 轨迹 | 最终 accepted | Valid Sharpe：Round 0 → 最终 | Test Sharpe：Round 0 → 最终 | 事后判断 | 改善来源 |
|---|---:|---:|---:|---|---|
| CSI1000-C2C | EXP_010 | 1.196 → 1.238 | 1.969 → 1.987 | 有限成功 | RAPA 将 turnover penalty 提高到 1.5 |
| CSI1000-O2O | EXP_009 | 1.312 → 1.374 | Round 0 报告缺失；EXP_009=1.074 | 失败倾向明确 | FAMA dropout=0.2 只在验证期更好 |
| CSI500-C2C | EXP_009 | 0.800 → 0.886 | 1.430 → 1.642 | 成功 | RAPA 将 turnover penalty 提高到 1.5 |
| CSI500-O2O | EXP_009 | 0.680 → 0.683 | 1.530 → 1.652 | 成功，但验证优势很小 | turnover penalty=1.3，risk aversion=1.1 |

整体上，三条轨迹取得测试期正向改善，其中两条改善较明确；CSI1000-C2C
只取得非常小的改善。CSI1000-O2O 出现明显的验证—测试错位。

## 3. CSI1000-C2C：组合层有限成功

### 逐轮过程

| 实验 | 层/动作 | Valid Sharpe | Gate 结果 |
|---|---|---:|---|
| EXP_000 | 五因子 Round 0 | 1.196 | BASELINE / accepted |
| EXP_001 | RASS：KUP + QTLD30 + CORD30 | — | CONTRACT_REJECTED（provenance hash 抄写错误） |
| EXP_002 | RASS：QTLD60 + VSTD60 + CORD30 | 1.133 | FALSIFIED |
| EXP_003 | RASS：ROC30 + IMIN60 + KUP | 1.259 | FALSIFIED（trade-off） |
| EXP_004 | RASS：KLEN + MA60 + CORD30 | 1.347 | FALSIFIED（trade-off） |
| EXP_005–007 | FAMA：dropout 0.0 → 0.2 | — | CONTRACT_REJECTED（路径格式错误） |
| EXP_008 | RAPA：turnover penalty 1.0 → 1.1 | 1.202 | PARTIALLY_SUPPORTED / accepted |
| EXP_009 | RAPA：1.1 → 1.3 | 1.221 | PARTIALLY_SUPPORTED / accepted |
| EXP_010 | RAPA：1.3 → 1.5 | 1.238 | PARTIALLY_SUPPORTED / accepted |

### Round 0 与最终结果

| 指标 | Round 0 Valid | Final Valid | Round 0 Test | Final Test |
|---|---:|---:|---:|---:|
| IC | 0.042 | 0.042 | 0.010 | 0.010 |
| RankIC | 0.036 | 0.036 | 0.017 | 0.017 |
| Sharpe | 1.196 | 1.238 | 1.969 | 1.987 |
| 年化收益 | 0.383 | 0.400 | 0.466 | 0.468 |
| 最大回撤 | -0.158 | -0.154 | -0.120 | -0.121 |
| 单边换手 | 0.059 | 0.045 | 0.060 | 0.045 |

最终模型和 Alpha 与 Round 0 相同；成功完全来自组合层。测试 Sharpe 仅增加
`0.018`，但方向与验证期一致，换手显著下降。因此应表述为“交易效率小幅改善”，
而不能表述为预测能力提高或强泛化成功。

## 4. CSI1000-O2O：验证改善没有泛化

### 逐轮过程

| 实验 | 层/动作 | Valid Sharpe | Gate 结果 |
|---|---|---:|---|
| EXP_000 | 五因子 Round 0 | 1.312 | BASELINE / accepted |
| EXP_001 | RASS：MIN20 + IMIN60 + CORD30 | 1.198 | FALSIFIED |
| EXP_002 | RASS：MIN60 + STD20 + CORD20 | 1.284 | FALSIFIED |
| EXP_003 | RASS：QTLD60 + IMIN10 + STD10 | 1.176 | FALSIFIED |
| EXP_004–006 | FAMA：dropout 0.0 → 0.2 | — | CONTRACT_REJECTED（路径格式错误） |
| EXP_007 | RAPA：turnover penalty 1.0 → 0.9 | 1.347 | FALSIFIED（trade-off） |
| EXP_008 | RAPA：1.0 → 1.1 | 1.283 | FALSIFIED |
| EXP_009 | FAMA：dropout 0.0 → 0.2 | 1.374 | PARTIALLY_SUPPORTED / accepted |
| EXP_010 | FAMA：dropout 0.2 → 0.3 | 1.274 | FALSIFIED；回滚 EXP_009 |

### 事后测试证据

Round 0 的 researcher test 在旧版 Python 3.9 `zip(strict=True)` 处失败，因此没有
生成完整的 Round 0 test `result.json`。但 EXP_007/008 只改变组合参数并复用 Round 0
预测，故其 IC/RankIC 可以作为基线模型预测的精确参照：

| 配置 | Test IC | Test RankIC | Test Sharpe | Test excess annual return |
|---|---:|---:|---:|---:|
| Round 0 模型预测（EXP_007/008 复用） | 0.026 | 0.043 | 1.204 / 1.100（penalty 0.9 / 1.1） | -0.138 / -0.151 |
| EXP_009，dropout=0.2 | 0.023 | 0.035 | 1.074 | -0.144 |
| EXP_010，dropout=0.3（被拒绝） | 0.024 | 0.033 | 1.127 | -0.142 |

EXP_009 在验证期将 Sharpe 从 `1.312` 提高到 `1.374`，但验证 IC/RankIC 已从
`0.041/0.047` 降到 `0.040/0.044`；测试期 IC/RankIC 又进一步低于基线模型，
测试 Sharpe 也低于两个相邻的基线模型组合对照。由此可判断 dropout=0.2 的
验证优势没有泛化。EXP_010 被 Gate 拒绝是正确方向，但回滚后的 EXP_009 仍不是
测试期更优配置。

## 5. CSI500-C2C：明确的组合层成功

### 逐轮过程

RASS 的有效执行候选均未通过 Gate，另外两轮因旧版 selected-features 使用因子 ID
而被合同拒绝。系统最终按规则回退五因子并冻结 Alpha。随后 RAPA 找到稳定方向：

| 实验 | 动作 | Valid Sharpe | Gate 结果 |
|---|---|---:|---|
| EXP_000 | Round 0 | 0.800 | BASELINE |
| EXP_001 | RASS 候选 | 0.801 | FALSIFIED |
| EXP_002 | RASS 候选 | — | CONTRACT_REJECTED |
| EXP_003 | RASS 候选 | 0.605 | FALSIFIED |
| EXP_004 | RASS 候选 | — | CONTRACT_REJECTED |
| EXP_005 | RASS 候选 | 0.780 | FALSIFIED；回退五因子 |
| EXP_006 | turnover penalty 1.0 → 0.9 | 0.803 | FALSIFIED |
| EXP_007 | 1.0 → 1.1 | 0.810 | PARTIALLY_SUPPORTED / accepted |
| EXP_008 | 1.1 → 1.3 | 0.847 | PARTIALLY_SUPPORTED / accepted |
| EXP_009 | 1.3 → 1.5 | 0.886 | PARTIALLY_SUPPORTED / accepted |
| EXP_010 | risk aversion 1.0 → 0.9 | 0.887 | FALSIFIED；回滚 EXP_009 |

最终相对 Round 0：

- Test Sharpe：`1.430 → 1.642`，增加 `0.212`；
- Test 年化收益：`0.259 → 0.329`；
- Test 单边换手：`0.040 → 0.030`；
- Test 最大回撤：`-0.101 → -0.106`，略有恶化；
- Test excess annual return：`-0.210 → -0.163`，仍然为负。

验证和测试对提高换手惩罚的排序一致，是本批四组中最清晰的成功案例。但成功来自
RAPA，而非 Alpha 或模型；绝对 Sharpe 改善也不等于获得正的基准超额能力。

## 6. CSI500-O2O：组合层成功，但验证优势接近阈值

### 逐轮过程

三次 RASS 候选全部被 Gate 否决并回退五因子。之后 RAPA 在换手惩罚和风险系数上
进行局部搜索：

| 实验 | 动作 | Valid Sharpe | Gate 结果 |
|---|---|---:|---|
| EXP_000 | Round 0 | 0.680 | BASELINE |
| EXP_001–003 | 三组 RASS 候选 | 0.725 / 0.757 / 0.740 | FALSIFIED；回退五因子 |
| EXP_004 | turnover penalty 1.0 → 0.9 | 0.671 | FALSIFIED |
| EXP_005 | 1.0 → 1.1 | 0.683 | PARTIALLY_SUPPORTED / accepted |
| EXP_006 | risk aversion 1.0 → 0.9 | 0.683 | FALSIFIED |
| EXP_007 | turnover penalty 1.1 → 1.3 | 0.684 | PARTIALLY_SUPPORTED / accepted |
| EXP_008 | turnover penalty 1.3 → 1.5 | 0.679 | FALSIFIED |
| EXP_009 | risk aversion 1.0 → 1.1 | 0.683 | PARTIALLY_SUPPORTED / accepted |
| EXP_010 | risk aversion 1.1 → 1.3 | 0.683 | FALSIFIED；回滚 EXP_009 |

最终相对 Round 0：

- Valid Sharpe：`0.680 → 0.683`，只增加 `0.003`；
- Test Sharpe：`1.530 → 1.652`，增加 `0.122`；
- Test 年化收益：`0.258 → 0.296`；
- Test 最大回撤：`-0.085 → -0.075`；
- Test 单边换手：`0.052 → 0.042`；
- Test excess annual return：`-0.191 → -0.165`，仍然为负。

测试结果支持 RAPA 的方向，但 validation 增益只比 `epsilon_sharpe=0.002` 高一点，
因此应称为“组合层的弱验证、正测试结果”，不能据此宣称参数在其他时间窗口稳定。

## 7. 跨四组结论

1. **三 seed ensemble 降低了训练随机性，但没有消除跨期失效。** CSI1000-O2O
   仍出现 validation 接受而 test 退化，说明主要问题不只是单 seed 噪声。
2. **RASS 在本批没有产生最终成功的八因子状态。** 四条轨迹最终均使用初始五因子；
   Gate 的拒绝与三次失败回退避免了把不稳定 Alpha 固化到最终状态。
3. **成功主要来自 RAPA。** CSI1000-C2C、CSI500-C2C、CSI500-O2O 都通过降低
   无效换手获得测试期改善，其中 CSI500-C2C 最清晰。
4. **绝对 Sharpe 与超额能力必须分开表述。** 两条 CSI500 轨迹虽然绝对 Sharpe
   提高，但最终测试 excess annual return 和 information ratio 仍为负。
5. **O2O 的模型参数跨期稳定性仍是主要风险。** CSI1000-O2O 的 dropout 选择在
   agent-valid 和 test 间反转，不能依靠更多同窗口搜索解决。

## 8. 本批运行中的机械性无效轮次

本批存在以下历史代码问题：

- FAMA 使用 `dropout` 或 `train_params/dropout`，而合同要求完整路径，导致
  `FAMA_CROSS_GROUP_PATH`；
- RASS 把因子 ID 而不是 Qlib expression 写入 `selected_features`，导致
  `RASS_APPEND_ONLY_TO_EIGHT_REQUIRED`；
- 一次 RASS provenance hash 被大模型抄错一个字符，导致
  `RASS_PROVENANCE_MISMATCH`；
- Python 3.9 不支持 `zip(strict=True)`，导致部分旧 researcher test 报告缺失。

这些问题均已在本批结束后改为确定性处理，并通过服务器完整测试 `98/98`。它们会影响
本批可用轮次数量，但不得据此重判或改写已经结束的轨迹。

## 9. 可用于论文的谨慎表述

> 在三 seed ensemble 设置下，DiagAgent 的组合层干预在三个任务中取得了验证与测试
> 同方向的绝对 Sharpe 改善，其中 CSI500-C2C 的改善最明确。系统能够拒绝多组不稳定
> Alpha 干预并回退到五因子基线，体现了跨层诊断、回滚与有界局部搜索的价值。
> 然而，CSI1000-O2O 中被验证接受的 dropout 配置在隔离测试期退化，说明 ensemble
> 只能降低优化随机性，不能解决验证窗口与未来市场阶段之间的配置排序反转。所有测试
> 结果仅用于事后研究分析，未参与正式轨迹的选择和接受。

