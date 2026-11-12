# 环境搭建实录（Windows 11，2026-11-12）

记录一次**实际成功**的搭建过程，包括踩到的坑。将来在 AutoDL 或换机器时照这个走，能省半天。

---

## 0. 结果

| 项 | 值 |
|---|---|
| SSIO 仓库 | `D:\MIU\SSIO` |
| Python | **3.11.9**（`D:\Python311`） |
| 虚拟环境 | `D:\MIU\SSIO\.venv` |
| torch | 2.5.1+cpu |
| pypose | 0.6.8 |
| 单元测试 | **39 passed, 4 skipped** |
| quickstart | 三步全通（训练→PGD→评测） |

---

## 1. 为什么不能用系统自带的 Python 3.13

本机原本只有 Python 3.13。两个硬冲突：

1. **`numpy<2` 在 3.13 上没有轮子**。numpy 1.26 最高支持到 3.12，3.13 只能装 numpy≥2.1。
   SSIO 的 `requirements.txt` 明确要求 `numpy<2`。
2. **torch 2.5.1 没有 cp313 轮子**。3.13 上最早能装的是 torch 2.6.0，而 README 说
   "tested with torch 2.5.1"。

所以装了 Python 3.11.9（国内镜像，26 MB，静默安装）：

```powershell
curl -L -o "$env:TEMP\py311.exe" https://mirrors.huaweicloud.com/python/3.11.9/python-3.11.9-amd64.exe
Start-Process "$env:TEMP\py311.exe" -Wait -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=0','Include_pip=1','TargetDir=D:\Python311'
```

> `InstallAllUsers=0` 不需要管理员权限；`PrependPath=0` 不污染系统 PATH——我们只在 venv 里用它。

## 2. 建虚拟环境

```powershell
cd D:\MIU\SSIO
D:\Python311\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**一定要用 venv**：这套依赖（numpy<2、torch 2.5.1）会覆盖系统里更新的版本，
装到全局会把你其他项目搞坏。

## 3. 分阶段装依赖（关键教训）

**不要一条命令装全部。** pip 是原子的：任何一个包下载失败，**整批都不装**。
我们第一次就是 wandb 下载被重置，结果连 torch 都没装上。

```powershell
$py  = '.\.venv\Scripts\python.exe'
$idx = 'https://pypi.tuna.tsinghua.edu.cn/simple'

& $py -m pip install "torch==2.5.1" -i $idx --retries 10 --timeout 60 --prefer-binary
& $py -m pip install "numpy<2" scipy matplotlib -i $idx --retries 10 --timeout 60 --prefer-binary
& $py -m pip install hydra-core omegaconf jaxtyping beartype rich h5py zarr pytest "huggingface_hub[cli]" -i $idx --retries 10 --timeout 60 --prefer-binary
& $py -m pip install "pypose==0.6.8" -i $idx --retries 10 --timeout 60 --prefer-binary
& $py -m pip install wandb -i $idx --retries 10 --timeout 60 --prefer-binary
& $py -m pip install warp-lang -i $idx --retries 10 --timeout 60 --prefer-binary
```

关于最后两个：

- **wandb** 是最容易下载失败的一个（包大、镜像偶发重置）。单独装，失败了重跑即可。
  训练命令里加 `wandb.mode=disabled` 就不需要联网。
- **warp-lang** 只被 `Optimizer/sparse_pgo.py` 用（CUDA 稀疏图）。Windows 上跑不了，
  但**不装的话 `pytest` 收集阶段就报错**，整个测试套件跑不起来。装上让它闲着即可。

## 4. 已知的版本"违规"（不影响使用）

装完后 `numpy` 实际是 **2.4.6**，违反了 `numpy<2` 的约束——后续某个包把约束顶掉了。
判断依据是官方测试：

```
39 passed, 4 skipped, 22 deselected
```

**测试全过**，说明 numpy 2.x 在这套代码上没有实际问题。**保留现状**，但要知道这是一处偏离；
如果将来出现诡异数值错误，第一个怀疑对象就是它。

## 5. Windows 专属注意事项

1. **必须加 `-X utf8`**，否则评测器打印指标名时会 `UnicodeEncodeError`：
   ```powershell
   .\.venv\Scripts\python.exe -X utf8 evaluate.py .\experiments\quickstart\pgd
   ```
2. **PowerShell 里 Hydra 的参数要单引号包住**：
   `'hydra.searchpath=[file://tests/fixtures/config]'`
3. 路径分隔符统一用 `/` 传给 Hydra。

## 6. 三步 quickstart（在仓库根目录）

```powershell
$py = '.\.venv\Scripts\python.exe'

# 1/3 积分自监督
& $py -X utf8 train_ssio.py network=airio_imu_only `
  'hydra.searchpath=[file://tests/fixtures/config]' `
  dataset@train_dataset=EuRoC_Smoke_Train dataset@eval_dataset=EuRoC_Smoke_Valid `
  train.epoch=1 train.batch=4 train.max_train_steps=2 train.dv_imu_input=raw `
  'meta.device=${as_device:cpu}' wandb.mode=disabled hydra.run.dir=./experiments/quickstart/ssio

# 2/3 位姿图蒸馏
& $py -X utf8 train_pgd.py network=airio_imu_only graph=dense_imupgo `
  'hydra.searchpath=[file://tests/fixtures/config]' `
  dataset@train_dataset=EuRoC_Smoke_Train dataset@eval_dataset=EuRoC_Smoke_Valid `
  train.epoch=1 train.batch=4 train.max_train_steps=2 train.dv_imu_input=raw `
  train.network_factor_sigma=1.0 `
  train.resume_from=file:./experiments/quickstart/ssio/checkpoint_00002.pt `
  'meta.device=${as_device:cpu}' wandb.mode=disabled hydra.run.dir=./experiments/quickstart/pgd

# 3/3 评测
& $py -X utf8 evaluate.py ./experiments/quickstart/pgd
```

`graph=dense_imupgo` 是 CPU 可用的稠密求解器，**只适合短轨迹**（quickstart 是 5 秒）。
真训练要用 `sparse_imupgo`——**那个只能在 Linux + CUDA 上跑**，这条限制直接决定了
W8 之后必须上 AutoDL 的 Linux 机器。

## 7. 把结果导出来喂给自己的评测器

`evaluate.py` 只打印指标、画图，**不落盘轨迹**。所以有了：

```powershell
D:\MIU\SSIO\.venv\Scripts\python.exe -X utf8 tools\dump_ssio_traj.py `
  D:\MIU\SSIO\experiments\quickstart\pgd --ssio-root D:\MIU\SSIO --out runs\quickstart
```

产出 `runs/quickstart/{gt,pred}_<seq>.csv`，然后：

```powershell
python -m ssio_eval.cli eval --est runs\quickstart\pred_MH_02_easy.csv `
  --gt runs\quickstart\gt_MH_02_easy.csv --align none --ave-frame world --ave-align none
```

**必须传 `--align none --ave-align none`** 才能和官方口径对齐，详见 `docs/METRICS.md` §4。

## 8. 换到 AutoDL 时怎么做

1. 选 **PyTorch 2.5.x + CUDA 12.x** 的镜像（自带 torch，省 200 MB 下载）
2. `pip install pypose==0.6.8 numpy<2 ...`（同样的分阶段策略）
3. Linux 上 `sparse_imupgo` 可用，但需要 `nvidia-cudss-cu12` + `nvmath-python[cu12]`
   （`requirements.txt` 里 `sys_platform == "linux"` 那两行，版本要按 CUDA 版本改）
4. 数据用 `DATA_ROOT=/root/autodl-tmp/datasets` 指过去