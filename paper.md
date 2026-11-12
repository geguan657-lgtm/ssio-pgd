# 论文骨架（只改不推翻）

> 规则：第 1 周就把这一页写完。之后每次有实验结论，回来改这里；**不许推翻重写**。
> 写不出来，说明方向还没想清楚，先去想，别去调代码。

## 暂定标题

*When Does Pose-Graph Distillation Help? Diagnosing and Stabilizing Self-Supervised Inertial Odometry*

## 一句话论断（英文，一句话，能背）

Pose-graph distillation in self-supervised inertial odometry degrades after a few rounds because each round imprints the student's own drift into the targets; anchoring, uncertainty-aware factor weighting, and validation-gated stopping keep the gain monotone across rounds.

## 为什么重要（3 句）

1. 无标签 IMU 训练让新平台免于采集真值轨迹，这是 SSIO 的核心卖点。
2. 但它的第二个阶段（PGD）**迭代几十轮就开始变差**，而论文只报告了最优轮次的结果，没有解释也没有修法。
3. 如果 PGD 的收益依赖一个没人调得准的停止点，那这个方法在真实部署里不可靠。

## 贡献（三句话版本）

1. **诊断**：三组受控实验，定位退化的来源（误差自我强化 vs 因子权重 vs 超参）。
2. **修法**：一组稳定化手段（锚定 / 不确定性加权 / 门控早停），单变量消融各自的作用。
3. **结论**：给出 PGD 什么时候有用、什么时候有害的可操作判据。

## 图 1（论文的脸）

横轴：PGD 轮次 0 / 1 / 10 / 40 / 80
纵轴：AVE（m/s）
两条线：原版（先降后升）/ 我们的（单调下降并收敛）

**这张图在第 3 周就能画出来（原版那条线）。第 3 周画不出来，整个计划要重排。**

## 表 1

| dataset | supervised | SSIO (no PGD) | +PGD (原版) | +PGD (ours) |
|---|---|---|---|---|
| EuRoC | 0.344 | ? | 0.401 | ? |
| TLIO | 0.084 | ? | 0.100 | ? |
| Blackbird | 0.362 | ? | 0.430 | ? |
| (第 4 个) | ? | ? | ? | ? |

指标：AVE（m/s，body 帧）；括号内附 ATE。

## 什么结果会推翻我

| 如果出现 | 说明 | 怎么办 |
|---|---|---|
| 调低蒸馏学习率后原版本来就不退化 | 退化只是超参问题 | 贡献改成「超参敏感性 + 自动选参」 |
| 三个修复都不优于「取第 1 轮的结果」 | 这个方向没有方法价值 | 转 Plan B：写诊断型论文 |
| 收益只在 1 个数据集成立 | 论断过宽 | 收窄到「行人 + 足式」并改标题 |

## 要引用的（先列 10 个，W11 再补全）

- SSIO (arXiv:2610.25988)
- AirIO (Ashesh et al.)
- TLIO (Liu et al. 2020)
- RoNIN
- Blackbird / MIPO / GrandTour 数据集论文
- IMU 预积分 (Forster et al.)
- 位姿图优化 / GTSAM
- 自监督蒸馏相关（如有）

## 目标场地与截稿

- **IROS 2027**：约 2027-03（以官网为准），6 页正文
- 备选：ICRA 2028（约 2027-09）、RA-L（滚动）
- 保底：ICRA/CVPR workshop（截稿多在 2–4 月）