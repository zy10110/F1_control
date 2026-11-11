# F1_control

F1 绳驱手 **URDF-A** 整手仿真与 G7 手套跟随。发布基线：2026-11-11。

当前链路：**HumanHand 已标定关节姿态 → 相对骨段角度 → 个人角度映射与连续修正 → F1 限位、屈伸耦合、拇指避让 → MuJoCo 显示**。腕部固定；不使用末端 IK 驱动这条实时链路。

## 安装与运行

Windows、Python 3.10 或以上，需要 Tkinter 和可用的 OpenGL 图形环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --check
.\.venv\Scripts\python.exe main.py --demo
```

`--demo` 不需要手套和个人档案，依次展示五指屈伸，并使用实时程序相同的拇指路径间隙检查。它是模型检查入口，不是手套跟随效果回放。

## 手套实时跟随

1. 在 Forsense HumanHand 打开原用户，使用已有官方标定，开启右手“相对手腕 21 点关节位姿”UDP，目标 `127.0.0.1:15001`。本工程也能将带前臂的 22 点包转换为同一手腕坐标接口。
2. 将与本版模型匹配的现有 F1 个人档案放到 `calibration/local/profile.json`。本机交付包已包含此文件；GitHub 仓库不包含个人档案和原始采集记录。官方手套标定与 F1 角度适配档案用途不同，不能互相替代。
3. 运行：

```powershell
.\.venv\Scripts\python.exe main.py
# 也可指定已有档案：
.\.venv\Scripts\python.exe main.py --profile "D:\profiles\profile.json"
```

### 左手手套镜像

左手手套可驱动同一只 F1 右手模型：HumanHand 开启左手“相对手腕 21 点关节位姿”UDP，收到的左手位姿按 `S = diag(1, -1, 1)` 镜像成等效右手位姿后进入同一条映射链路。左手使用独立的个人档案（档案内 `input_hand` 为 `left`），默认读取 `calibration/local/profile_left.json`，手别与档案不符时拒绝启动。

```powershell
.\.venv\Scripts\python.exe main.py --hand left
.\.venv\Scripts\python.exe main.py --hand left --check
```

左手窗口提供“用官方手模标定（八个动作）”入口：依次采集张开、握拳、重复握拳、拇指活动及拇指分别贴四指，按 HumanHand 解算出的骨段方向和指尖间距在原限位内修正档案，再用已有的“留缝擦过 / 轻碰再松开”记录回放对比。窗口底部实时显示各指“手套指尖 / F1 指腹”到拇指的距离（mm）。

一个端口同时只运行一个接收实例。拖动模型旋转视角，滚轮缩放；数据中断超过配置阈值后保持当前姿态，不自动回零。`results/urdf_a_live_status.json` 提供输入序号、目标/实际角度、档案哈希、间隙及避让状态。

在 PyCharm 中打开本目录，选择项目虚拟环境并安装依赖，运行 `main.py`。随包提供 `F1 URDF-A` 运行配置；离线查看时添加参数 `--demo`。

## 模型与边界

- 当前几何：`assets/urdf_a/urdf/F1绳驱手-URDF-A.urdf`，38 个 link、37 个原生转轴、17 个独立整手控制量；两根腕部转轴固定。
- 每个屈伸连接处，上下两个原生转轴各承担语义关节角的 1/2。四根长指保持 `DIP = 0.8 × PIP`，拇指 MCP/IP 分别控制。
- 独立角度边界由相关原生转轴限位求交得到。几何和耦合哈希必须与个人档案一致；更换模型后不自动套用旧档案。
- 拇指避让沿运动路径按不大于 0.75° 的增量采样，监测拇指与掌部、长指的指定碰撞网格对，间隙阈值 0.6 mm。它不是全手连续碰撞保证。
- 当前用 `q_native = C @ u` 执行运动学耦合，未通过约束绳、张力或接触动力学实现机械耦合。该版本不输出硬件指令，也不用于验证真实绳路或舵机性能。

`assets/` 中的旧版模型及 `calibration/` 下的固定几何距离表供既有拟合工具使用；实时入口始终加载 URDF-A。`tools/` 保留拟合与回放的依赖代码，重新拟合还需要原训练数据及对应旧档案，本仓库未包含这些个人资料。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py --check --render
```

模型校验清单：`docs/model_sha256.json`。发布检查结果：`docs/validation.md`。

## 数据与分发

个人标定、原始录制、日志、厂家 SDK/可执行文件均在 `.gitignore` 中排除。仓库包含 F1 模型与本地开发代码，未附加开源授权；使用、修改及再分发遵循各自权利人的授权。
