# ssio-pgd

复现 **SSIO（[arXiv:2610.25988](https://arxiv.org/abs/2610.25988)）** 并修 PGD（位姿图蒸馏）的迭代退化。
12 周计划见 [docs/PLAN.md](docs/PLAN.md)。

> **这是复现脚手架，不是 SSIO 官方代码。**
> 官方实现：[amazon-far/SSIO](https://github.com/amazon-far/SSIO) ｜ 项目主页：[ssio-imu.github.io](https://ssio-imu.github.io/)
> 本仓库只包含评测、数据校验和实验记录模板，不含原作者的训练代码。

本目录当前状态：**评测脚手架已就绪并通过自测**；SSIO 代码集成尚未开始（W1 的活）。

---

## 30 秒跑通

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_baseline.ps1
# 装了 PowerShell 7 的话：pwsh -File scripts/run_baseline.ps1
```

它依次检查：Python 依赖 → 19 个单元测试 → 已知答案自测 → 合成数据端到端 → （有数据时）数据集校验。
全绿说明**评测器可信**，可以开始跑模型了。

## 三条命令

```powershell
# 1. 生成一对“已知答案”的合成轨迹（不需要任何数据集）
python -m ssio_eval.cli synth --out runs\synth

# 2. 评测：ATE / AVE / RPE
python -m ssio_eval.cli eval --est runs\synth\est.csv --gt runs\synth\gt.csv `
  --align se3 --ave-frame body --rpe --plot runs\synth\plot.png --out runs\synth\report.json

# 3. 批量评测，输出论文表格那种 CSV
python -m ssio_eval.cli batch --config configs\datasets.yaml --out runs\batch.csv
```

PowerShell 里如果没有设置过路径：

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONUTF8 = "1"
```

## 目录

```
ssio-pgd/
├── src/ssio_eval/          评测库
│   ├── frames.py           SO(3) / 四元数（无 scipy 依赖）
│   ├── traj.py             轨迹容器：重采样、插值、速度
│   ├── metrics.py          ATE / AVE / RPE / 对齐 / 时间偏移扫描
│   ├── io/tables.py        表头嗅探 + 列解析（EuRoC/TLIO/Blackbird 通吃）
│   ├── io/datasets.py      数据集布局
│   ├── synth.py            已知答案的合成数据
│   └── cli.py              eval / batch / synth / selftest
├── scripts/
│   ├── run_baseline.ps1    一键体检
│   ├── verify_data.py      数据体检（单位/丢帧/四元数）
│   └── get_data.py         下载或导入已下载的数据
├── tests/                  19 个已知答案测试
├── configs/                评测配置 + 下载清单
├── docs/METRICS.md         ← 报数字之前必须读
├── docs/DATA.md            ← 下数据之前必须读
├── paper.md                论文骨架（第 1 周就要写）
├── baseline.md             复现基线（W4 冻结）
├── diagnosis.md            W5 诊断报告
├── experiments.md          实验日志
└── weekly/                 周报
```

## 已经验证过的

- 恒等轨迹 → ATE / AVE 精确为 0
- 已知刚体变换 → `se3` 对齐后 ATE < 1e-9
- 已知纯缩放 → 只有 `--scale` 能吸收
- 已知常值速度偏差 0.0539 m/s → body 帧 AVE 复现为 0.0539
- 纯航向旋转 → `yaw` 对齐能还原出角度本身
- 合成数据写出 EuRoC 风格 CSV → 读回 → 评测，全链路闭环

## 还没做的（别误会）

- **没有接入 SSIO 官方仓库**。它的输出格式要等 W1 跑通才知道，届时在 `runs/` 下整理成 CSV 即可，评测侧不用改。
- **不读 IMU**。评测只用轨迹；IMU 是训练的事。
- **不支持 `.h5`**（MIPO/GrandTour 可能是这个格式）。到 W9 再按实际格式加一个 adapter。
- **没有时间戳对齐的自动关联**（比如 est 和 gt 采样率不同且都有时间戳时，我们按 gt 时间戳插值，不做最近邻匹配）。

## 下一步

1. 读 `docs/DATA.md`，下 EuRoC 的 5 个序列；
2. 读 `docs/METRICS.md`，知道每个数字的口径；
3. 按 `weekly/W01.md` 做，W1 结束时你要能跑通官方 inference 并画出第一条轨迹。

## License

MIT — 见 [LICENSE](LICENSE)。评测代码可自由复用；论文与数据集的版权归各自作者。