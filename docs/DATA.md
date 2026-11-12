# 数据准备（W1 的活）

## 1. 需要的目录长什么样

加载器只认这个结构（`configs/download.yaml` 里有同样的注释）：

```
data/euroc/MH_02_easy/
  mav0/state_groundtruth_estimate0/data.csv    <- 真值（必须）
  mav0/imu0/data.csv                           <- IMU（训练用）
  mav0/cam0/data.csv                           <- 相机时间戳（可选）

data/tlio/subset/
  gt/data.csv
  imu0/data.csv
```

EuRoC 解压后本来就是这个结构（`mav0/...`），**不要改名**，直接把整个序列文件夹拖进 `data/euroc/` 即可。

## 2. 从哪拿

| 数据集 | 来源 | 规模 | 什么时候需要 |
|---|---|---|---|
| EuRoC MAV | ETH ASL 官方页面（搜 "EuRoC MAV Dataset"），下载 `MH_02_easy` 等 5 个 zip | 每个 ~1–3 GB | **W1** |
| TLIO | TLIO 论文主页 / 其 GitHub（需填表或走 Google Drive），先取 2–3 小时子集 | 全量约 200 GB | **W2–W3** |
| Blackbird | MIT AERA 的 Blackbird 数据集主页（Google Drive） | 中等 | W9 |
| MIPO | MIPO 论文主页 | 中等 | W9（备选） |
| GrandTour | ANYmal GrandTour 数据集主页 | 大 | 暂不需要 |
| Humanoid | 论文作者自采（Unitree G1） | — | **不做**，你没有硬件 |

> 这些下载入口经常变（表单、网盘、注册）。**用浏览器下，再用 `scripts/get_data.py --from-dir` 导入**，比试图写死 URL 稳得多。

## 3. 导入与校验

```powershell
# 浏览器下载到默认下载目录后
python scripts/get_data.py --from-dir "$env:USERPROFILE\Downloads" --dest data/euroc

# 校验：布局、时间单位、丢帧、四元数范数
python scripts/verify_data.py --root data/euroc --format euroc
```

正常输出长这样：

```
sequence                         n   dur[s]  dt[ms]   path[m]  gaps  status
------------------------------------------------------------------------------
MH_02_easy                   21000    180.0    5.00     400.1     0  OK

1 sequence(s) usable, 0 unusable.
```

看到下面任何一条，**先修数据再训练**：

| 现象 | 原因 | 处理 |
|---|---|---|
| `implausible duration 1.4e9s` | 时间戳单位是 ns，被当成 s | 用 `--time-unit ns`，或检查 `detect_time_unit` |
| `quaternion norm deviates by 2e-2` | 四元数没归一化 / 列错位 | 归一化；核对表头 |
| `3 gap(s) > 10x median dt` | 真丢帧 | 记录下来，评测时注意 |
| `no orientation columns` | 文件没有姿态列 | body-AVE 不可用，改用 world |
| `only 1 column parsed` | 分隔符判断错（空格 vs 逗号） | 看表头，必要时手动指定 |

## 4. 磁盘与时间预算

- EuRoC 5 个序列：约 **8–12 GB**
- TLIO 子集 2–3 h：约 **5–10 GB**（全量别现在下）
- 预算：**留 40 GB 空闲**

解压很慢（几万个文件）。**下载和解压放在晚上挂机做，不要占用白天写代码的时间。**

## 5. 一个容易忽略的点

EuRoC 的 `data.csv` 第一行是 `#timestamp, p_RS_R_x [m], ...` 这种**带单位的表头**。
我们的解析器会去掉 `[...]` 再匹配列名；如果你自己写解析，注意别把 `[m]` 当成列名的一部分。

## 6. 下一步

数据 verified 通过后，去 W1 的目标：**用官方仓库跑通一次 inference**，把输出整理成
`<timestamp>, p_x, p_y, p_z, q_w, q_x, q_y, q_z` 这种最小格式（或直接保留原格式），
然后：

```powershell
python -m ssio_eval.cli eval --est runs\w1_demo\MH_02_easy.csv `
  --gt data\euroc\MH_02_easy\mav0\state_groundtruth_estimate0\data.csv `
  --align se3 --ave-frame body --plot runs\w1_demo\MH_02_easy.png
```

**W1 的验收标准不是数字好看，而是这条命令能跑通并画出轨迹。**