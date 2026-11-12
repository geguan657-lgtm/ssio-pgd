# 实验日志

> 规则：**开卡之前**写一行假设，跑完写一行结论。没写的实验等于没做。
> 周五花 10 分钟扫一遍这张表，你会发现自己少跑了哪些对照组。

| # | 日期 | 假设（一句话） | 配置 | 数据集 | 结果 | 结论 | 下一步 |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-20 | 官方 demo 能直接跑通 | default | MH_02 | | | |
| 2 | | | | | | | |

---

## 模板（复制这段）

```
### EXP-000 短标题
- 日期：
- 假设：如果我做 X，那么 Y 会变化，因为 Z。
- 配置：configs/exp000.yaml ｜ commit ｜ 种子
- 命令：`python ...`
- 成本：__ GPU·h（¥__）
- 结果：表/图路径
- 结论：支持 / 不支持假设（一句话）
- 意外发现：
- 下一步：
```

---

## 常用命令

```powershell
# 单条评测
python -m ssio_eval.cli eval --est runs\<run>\<seq>.csv --gt data\euroc\<seq>\mav0\state_groundtruth_estimate0\data.csv `
  --align se3 --ave-frame body --plot runs\<run>\<seq>.png --out runs\<run>\<seq>.json

# 批量（论文表格）
python -m ssio_eval.cli batch --config configs\datasets.yaml --out runs\<run>\batch.csv

# 时间偏移诊断（怀疑时间戳没对齐时）
python -m ssio_eval.cli eval --est ... --gt ... --offset-scan 0.5
```

> `--offset-scan` 只是诊断工具。**除非官方评测也这么做，否则不许把扫描后的数字写进论文。**

---

## 对照组清单（每次跑主实验都要问一遍）

- [ ] 监督基线（同结构、有标签）——这是论文的分母
- [ ] 无 PGD 的自监督（第 0 轮）
- [ ] 原版 PGD（第 1/10/40/80 轮）
- [ ] 我们的修复（单变量）
- [ ] 3 个种子，报均值±标准差
- [ ] 参数量 / 训练时间 / 推理时间（审稿人现在越来越爱问）