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

## 2.5 2026-11 实测：官方数据现在只剩 bag 包，但里面藏着原始 zip

这一条是踩了整整一轮才发现的，**记下来能省你一整天**。

**现状**：`robotics.ethz.ch` 已下线（DNS 能解析，TCP 全超时）。官方数据迁到了
ETH Research Collection（DOI [10.3929/ethz-b-000690084](https://doi.org/10.3929/ethz-b-000690084)），
但那里只发布 4 个文件：

```
machine_hall.zip     12.7 GB
vicon_room1.zip       6.0 GB
vicon_room2.zip       6.0 GB
calibration_datasets.zip  4.4 GB
```

**关键发现**：这三个大包里，每个序列目录下**同时**放了两样东西：

```
machine_hall/
  MH_02_easy/
    MH_02_easy.bag    2.2 GB   ← ROS bag，不是我们要的
    MH_02_easy.zip    1.3 GB   ← 官方原始 mav0/ 结构，要的就是它
```

所以**不需要做 bag 转换**，直接解开内层 zip 即可。验证方法：解开后与 SSIO 仓库
自带的 `tests/fixtures/euroc_mh02_10s`（作者从同一 DOI 切的 10 秒样本）逐值比对，
IMU 和 GT **完全一致**。

**操作**（PowerShell，只取 CSV/YAML，跳过 90% 体积的图像）：

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$outer = [System.IO.Compression.ZipFile]::OpenRead('machine_hall.zip')
$entry = $outer.Entries | Where-Object { $_.FullName -eq 'machine_hall/MH_02_easy/MH_02_easy.zip' }
[System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, 'MH_02_easy.zip', $true)
$outer.Dispose()

$z = [System.IO.Compression.ZipFile]::OpenRead('MH_02_easy.zip')
foreach ($e in $z.Entries) {
  if ($e.FullName -notmatch '\.(csv|yaml)$' -or $e.FullName.EndsWith('/')) { continue }
  $t = Join-Path 'data\euroc\MH_02_easy' $e.FullName.Replace('/','\')
  New-Item -ItemType Directory -Force -Path (Split-Path $t) | Out-Null
  [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $t, $true)
}
$z.Dispose()
```

每个序列解出 **11 个文件、约 8 MB**（图像在 zip 里但不解）。

**下载**：官方站点在国内直连不通（Research Collection 会返回 429 / Access Restricted），
用 HuggingFace 镜像 `GlowBond/EuRoC_MAV_Dataset`——**文件名和大小与官方完全一致**，
是同一份文件的副本：

```powershell
powershell -File scripts\fetch_parallel.ps1 `
  -Url 'https://hf-mirror.com/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main/machine_hall.zip' `
  -Out data\raw\machine_hall.zip -Connections 4
```

> **并发数选 4，不要选 16。** 实测：4 连接 52 MB/s，16 连接被限速到接近 0。
> 12.7 GB 大约 5 分钟；单连接只有 1–3 MB/s，要 3 小时。

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