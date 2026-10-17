# 任务一：ACT 视觉与运动轻量修正

使用原任务一 ACT 权重和控制流程，在抓取、称重和交接事件加入视觉检查、小幅关节修正及夹爪时序控制。不训练，不覆盖原模型、配置或入口，不使用物体真值。当前实测状态见 [HANDOFF.md](HANDOFF.md)。

## 保留的默认录制配置：仅首次抓取辅助，之后 ACT

历史录制对照选择 `configs/recording_first_grasp_then_act.yaml`，并已同步为 `config.yaml`。首次抓取确认成功后的下一控制步，永久关闭自定义视觉/称重/再抓/交接控制与夹爪保护、预测刷新；只保留通用关节限位和限速。日志切换标记 `ACT_DIRECT_AFTER_FIRST_GRASP`。

纯 ACT 对照配置为 `configs/recording_pure_act.yaml`，用 `--mode baseline` 启动。两遍均须从完整重启的 seed 0 场景开始。

## 2026-10-17 续研状态：首包结束后全部视觉控制

v63第三次同版本独立复位20261016-212331已确认完整四包通过，前三轮连续成功（38.88/38.40/40.47分钟）。随后第一次正式录像20261017-011742在首包原落稳检查失败，第二次录像20261017-012700完整四包通过：12487步/控制46.01分钟，四包2799/5879/9013/12487回位，24必要开闭、ACT回接0、搜索0。正式完整原速第三视角MP4已交付：results/videos/v63-four-parcels-third-view-20261017.mp4，46分58.5秒、1280×720、30fps、84555帧，逐帧解码及连续时间戳通过，无损封装视频流哈希一致。332测试/115冻结文件不变。v63累计5次独立运行4次完整成功、1次失败，不能宣称每次稳定或官方裁判成功；task_success=null。所有控制器、反馈记录器、录像和官方sim71均已停止，两容器仅入口shell。

下一阶段按用户要求提升实际执行速度与稳定性、开展多 seed 验证，并在稳定提速后另录完整原速第三视角视频。当前只有 seed 0，尚未开始非 0 seed 或提速实验；优先处理首次录像首包落稳候选污染问题，再按阶段耗时优化。已核实官方 `--seed` 接口；构建阶段 `--no-randomize` 不应自行修改。具体顺序、边界和实测基线见 [最新交接文档](HANDOFF.md)。

[完整第三视角视频](results/videos/v63-four-parcels-third-view-20261017.mp4) · [录像轮复盘](results/full-visual-v63-video-review-20261017.html) · [全部v63独立运行记录](diagnostics/full-visual-v63-all-attempts-20261017.json)

最新路线使用 `configs/visual_cycle_four_parcels.yaml`：首包沿用 ACT 与既有视觉辅助完成抓取、称重和再抓；再抓后由 `VisualCycle` 完成交接、入框、返回。首包结束后启用 `VisualPickup`，第二至第四包的选择、接近、抓取、称重、再抓和完成段全部冻结 ACT。默认录制配置保持不变，需要明确选择实验配置。

原 v15 首包配置已经通过三次相同代码、独立完整 seed 0 复位（094728 / 095914 / 104115），每次恰好 6 次必要夹爪开闭。四包实验 105713 在第一包完成后恢复 ACT 时又朝首包空位闭爪；清空历史后的只读新预测也出现同方向选包，不能只归因于融合。因此按照用户最新要求，首包以后不再恢复 ACT。

新增视觉拾取使用 RGB-D 桌面范围和尺寸筛选剩余包裹，复用首包请求的抓取目标。125016/v21、140152/v25、142846/v26分别独立连续完成前两包，各恰好12次必要夹爪开闭，首包后零ACT回接；版本不同，不能算同版本重复验证。前两次第三包称重准备触及腕限位，第三次停在空爪接近保护，均未完成四包。后续包加深10mm在当时第二包得到约21mm插入及28mm落稳余量；第三包后来的实际进展见下方当前结果。

152456/v31恢复原34mm释放高差并仅补偿首包X方向最多8mm后，独立完成两包，首包/第二包落稳余量21.9/28.1mm；首包新设置仍只有一次完整通过，不能宣称已可重复。第三包6063在转头期间位移保护停。v32先等相机到位再锁定目标，同现场续测154405通过早期接近，但下降时再度位移保护，尚未闭爪。只读四条桌面边界回放显示两个停止样本扣除共同坐标变化后目标位移约6mm。

此前 v47 保留原抓取、称重、落稳、再抓检查和首包后 ACT 冻结。v36–v38 改进真实双边缘与再抓近侧截面；v39 以至少三条真实屋顶边测量接近中的物体运动；v41 将后续包加深请求从 10 改为原允许的 15 mm。v40 的 36 mm 准备间隙/.18 姿态权重试验失败并已撤回，当前称重准备仍为原 30 mm/.12。

此前独立运行 `20261015-202310`（v41）在 2822/6144/9372 步完成前三包，恰好 18 次必要夹爪开闭、首包后 ACT 回接 0；第四包空爪接近时桌面参照确认失败，9475 步停止。原始 RGB-D 证明四条桌边仍可见，但凸包简化丢失了一边的中间采样，支撑跨度被低估。v42 保留原矩形、四边跨度与位移门槛，用实际桌面到低处的深度边界采样确认跨度，排除手部和图像边缘。当时 v47 的 267 项测试通过，83 个清单文件与容器及快照一致。

历史 v43 `20261015-211734` 完成前三包，第四包称重释放和原落稳通过（支撑余量18.12mm），再抓调平在原±30°修正范围内未收敛，11391步停止，未再抓闭爪。v43 支撑优先排序得到一次第四包落稳证据，**四包尚未通过**。v43第二次在6110步第三包接近停止，真实RGB-D定位为相邻长边点混入短边关联。v44按真实边界连续组分离，保留原三边与10mm保护，261项测试通过；第四包调平问题仍待验证，不扩大限幅或降低检查。[历史 v43 复盘](results/full-visual-v43-review-20261016.html)与[当前交接文档](HANDOFF.md)保留失败证据。通过后仍须同版本完整复位重复验证。

完成段使用 RGB-D、相机标定、公开 URDF 与夹爪反馈。左爪稳定接取后才松右爪，完整框口足印通过后才松左爪；新增框内点云稳定后返回准备位。原 `residual.py`、`settlement.py`、`wrist.py`、默认 `config.yaml` 保持字节不变。模型权重未改，这些视觉控制结果不表示 ACT 新预测或融合已经修复。原环境 `task_success=null`，视觉确认不替代官方任务判定。

`--visual-cycle-from` 用于同现场中断的完成段；`--visual-next-from` 只接受已完成一包并回位、双手为空的现场；`--visual-weigh-from` 使用精确且未过期的持物检查点，先重新确认姿态，再进入称重前准备。这些入口校验 ROS 场景和物理状态，只用于 `--task run`，续测不计独立成功。`go_run` 只允许官方场景完整复位后使用；`--stop-after-parcels 2` 可在四包布局中先验证到累计第二包。

[上一轮两次首包复盘（历史报告）](results/visual-cycle-review-20261015.html)；[三次首包审计](diagnostics/visual-cycle-three-independent-20261015.json)。历史原生 ACT 预测诊断工具保留；hook 前后动作完全一致，但没有证明模型后续阶段可靠。

## 历史完整视觉辅助方案（旧录制对照配置关闭）

`config.yaml` 使用 `cartesian_alignment.strategy: act_residual` 和 `motion.direct_to_wbc: false`，保留原 MPC 控制通道与原执行链路。ACT 图像保持 `[3,480,848]`，归一化和动作反归一化不变。动作布局为左臂 7 关节、左爪、右臂 7 关节、右爪。

- 首抓、称重放置和再抓使用视觉辅助；再抓接触确认后，由 ACT 控制双臂完成交接与入箱。抓取和释放事件短暂冻结策略，RGB-D 与公开 URDF 局部 Jacobian 提供纠偏；常规残差上限 15°，称重调平及空爪再抓为已授权的 30°，同时保留实际 URDF 限位和每步增量上限 0.012 rad。
- 抓取前确认包裹目标与实际末端位置，原闭爪值 0.25–0.6 时保持张爪并让 ACT 继续下降，0.6 时才冻结做 XYZ 小幅对齐；抓取高度以包裹中心计算，Z 误差 ≤10 mm 才闭爪。闭爪时以视觉抓取目标（不是之前偏低的测量位置）做局部反馈，稳定接触后再继续抬起。
- 称重检查在原 ACT 渐进张爪丢失夹持力之前介入。头部先左右观察，再看包裹和称重台；视觉定位台中心与完整边界，并检查包裹足印。
- 放置目标约 10°、允许松爪倾角不超过 15°，最低边间隙不超过 2 mm，底面高度差不超过 34 mm；检查足印及末端稳定后才松爪。之后短退、视觉确认落稳、右腕外轮廓对中并稳定闭爪。
- 头部在称重释放、落稳和视觉再抓期间持续看称重台；再抓完成后保持手臂，恢复 ACT 的 `[0°,20°]` 头部视角再恢复推理。再抓只要求右腕边缘证据，不要求头部看到包裹。
- 交接保护仅在左爪强闭合且受阻反馈稳定后允许右爪张开。新阶段衔接试验会在当前预测要求闭爪、历史融合要求张爪连续冲突时刷新一次历史预测。该条件的从头实测结果见 HANDOFF；单独锁右爪曾导致停滞。原模型、输入和双臂动作均来自 ACT。
- 首包入箱已有用户确认。入箱后的下一包视觉辅助状态复位及四包连续循环尚未通过，不能把首包成功当整任务成功。

头部左右限幅 ±70°、俯仰 ±25°并限速。ACT 冻结时改变视角，恢复原视角并稳定后再推理。模型输入图像不画框、不裁切，旁路 RGB-D 处理不修改输入。

旧 `scene1/cartesian.py` 的分段重规划保留供历史排查，默认不使用。现场恢复入口必须匹配其历史日志和当前夹爪/关节状态；不能把现场续接作为独立成功次数。

## 文件

| 文件 | 用途 |
| --- | --- |
| `scene1/residual.py` | ACT 局部视觉残差、调平和夹爪事件检查 |
| `scene1/rgbd.py` | RGB-D 同步、包裹候选和台面边界 |
| `scene1/kinematics.py` | 实际公开 URDF 的末端 FK 和局部 Jacobian |
| `scene1/head.py` | 左右观察、限速与恢复原 ACT 视角 |
| `scene1/motion.py` | 动作限速与原准备状态处理 |
| `scene1/visual_cycle.py` | 再抓后的视觉交接、入框、落稳验证与回准备姿态 |
| `scene1/visual_pickup.py` | 首包后视觉选择剩余包裹、接近、试抬、搬运与称重前准备 |
| `scene1/receiver.py` | 接收手腕 RGB 真实边缘对中，包含同材质阴影端面 |
| `scene1/pickup_resume.py` | 同现场持物称重诊断的精确检查点与过期校验 |
| `scene1/handover.py` | ACT 交接夹爪保护及历史预测冲突检测 |
| `scene1/policy_trace.py` | 当前预测与历史融合对照，不改模型输出 |
| `scene1/trace_window.py` | 有上限的连续原始输入与完整 chunk 归档 |
| `scene1/handover_contract.py` | 可选的有限时交接反馈检查与接管缓速实验 |
| `diagnostics/probe_act.py` | 无运动采样和原模型离线预测 |
| `diagnostics/analyze_handover.py` | 旧日志统计，不把稀疏样本当完整融合回放 |
| `diagnostics/replay_trace.py` | 新格式完整 chunk 的原生融合回放和同输入模型检查 |
| `run_scene1.py` | 原 bag 回放、ACT 推理和日志 |
| `inspect_rgbd.py` | 无运动 RGB-D 与 FK 检查 |
| `config.yaml` | 优化参数，不覆盖原场景配置 |

宿主机项目 `/home/krist/kuavo_course/scene1_act_optimization`，算法容器 `kuavo-ros-noetic-s12-ready` 内 `/root/scene1_act_optimization`。原算法目录 `/root/LeTools-Learning` 保留。仿真容器 `kuavo_challenge_container_GPU_046e4301` 使用官方入口启动。

## 同步和运行

当前版本和运行状态以 [HANDOFF.md](HANDOFF.md) 为准，旧追加历史已独立备份。同步只在控制器停止时进行。

停止控制进程后，将已验证的宿主修改同步到容器：

```bash
cd /home/krist/kuavo_course/scene1_act_optimization
bash sync_to_container.sh
```

同步只包含本项目代码、测试、配置和文档，末尾执行无运动 preflight。

仿真容器沿用教程启动原始场景：

```bash
cd /root/kuavo_ws
source installed/setup.zsh
source devel/setup.zsh
export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH
rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0 --no-timer-gui
```

宿主机运行算法：

```bash
# 无运动检查
sudo docker exec kuavo-ros-noetic-s12-ready bash /root/scene1_act_optimization/run.sh --preflight

# 完整复位后的连续试验
sudo docker exec -it kuavo-ros-noetic-s12-ready bash /root/scene1_act_optimization/run.sh --config configs/visual_cycle_four_parcels.yaml --task go_run --mode optimized --stop-after-parcels 4 --max-steps 16800
```

`go_run` 回放原 go.bag 并执行策略；机器人已准备好时可用 `run`。`baseline` 和 `observe` 也会发送原 ACT 动作，只有 preflight 不运动。不要同时运行多个控制程序，不在持物状态重新启动会张爪的初始化入口。每次独立试验先真实复位场景，连续调试不能计为重复成功。

运行入口的 SIGUSR1 暂停/恢复策略发送，SIGUSR2 停止；这不是仿真复位或暂停服务。停止后保留最后指令，不额外回零或张爪。

## 验证与日志

```bash
cd /root/scene1_act_optimization
/root/miniforge3/envs/kls_dev/bin/python -m unittest discover -s tests -v
bash run.sh --preflight
```

改动前历史记录是 143 项离线测试通过，旧诊断记录器曾以原模型验证输出完全一致。该结果不覆盖本轮改动；当前验证范围见上方续研说明。所有离线测试都不能替代实际取放结果。

容器 `outputs/<时间>/` 保存 `steps.jsonl`、配置、动作、关节反馈、RGB-D 候选、头部阶段、视觉放置检查与三相机图片。`summary.json` 的 `task_success` 保持 null，因为原环境没有提供任务结果判定。`release_completed` 和称重台变色不能证明包裹稳定留在台上；需检查实际包裹方向、边距和静止情况，再验证再抓取、交接和入箱。

最新v44 `20261015-223226` 在685步首包原称重期限停止，未释放，未到第三包边界修正。v45控制逻辑同v44，仅启用每20步原始RGB-D/两腕及视觉持物估计记录，用于追查首包倾斜与第四包再抓跟踪问题；不扩限幅或放宽检查。详见 [v44复盘](results/full-visual-v44-review-20261016.html)。
