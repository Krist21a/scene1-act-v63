# Scene 1 ACT 视觉辅助四包执行：冻结 v63

这是 2026-10-17 保存的 **v63 历史成功版本**。115 个代码、配置和测试文件逐字来自冻结快照，SHA-256 与原清单一致。后续 v64 研发代码未收录。

首包通过原 ACT 与视觉辅助完成初抓、称重、落稳、再抓，随后视觉完成交接、入箱和回位；第二至第四包完全使用视觉控制。运行必须显式指定 `configs/visual_cycle_four_parcels.yaml`。默认 `config.yaml` 保留旧录制对照，不能用它替代四包配置。

## 已验证的实际结果

同版、官方完整复位、seed 0 共五次：四次完整四包完成，一次首包落稳失败。所有记录的 `task_success=null`，不是官方裁判通过证明；尚未验证非 0 seed。

| 运行 | 结果 | 步数 | 实际控制时间 |
| --- | --- | ---: | ---: |
| 20261016-200019 | 四包完成 | 12426 | 38.88 分钟 |
| 20261016-204228 | 四包完成 | 12271 | 38.40 分钟 |
| 20261016-212331 | 四包完成；第四包一次受限搜索 | 12832 | 40.47 分钟 |
| 20261017-011742 | 首包落稳失败，保护阻止再抓 | 最后发送 527 / 异常观测 528 | 未完成 |
| 20261017-012700 | 正式录像，四包完成 | 12487 | 46.01 分钟 |

[全部尝试总账](diagnostics/full-visual-v63-all-attempts-20261017.json)保留全部成功与失败。原 332 项测试通过；本次独立导出验证结果见 [重新测试日志](evidence/v63-github-export-tests-20261017.txt)。

## 完整原速录像与日志

- [成功四包第三视角 MP4](results/videos/v63-four-parcels-third-view-20261017.mp4)：46:58.5、1280×720、30 fps、84555 帧。录制始于 `go_run` 前，覆盖四包最终回位，原速连续，无剪接加速。
- [首次失败录像](results/videos/v63-video-attempt1-FAILED-20261017.mkv)同样保留。
- [成功录像审计](diagnostics/v63-formal-video-session-20261017.json)与[成功复盘](results/full-visual-v63-video-review-20261017.html)。
- 五次运行的 `run.json`、`summary.json` 和**完整** `steps.jsonl.gz` 位于 `results/<run>/`；压缩前后 SHA 已核对。执行 `gzip -dk results/<run>/steps.jsonl.gz` 可恢复逐步明文日志。
- 首次失败的七份关键原始 RGB-D NPZ 已收录。其余全量周期图像、RGB-D、辅助 ROS 反馈仍在原工作站，本仓库保留[原始结果逐文件 SHA 清单](evidence/original-results-manifest.json)，不是全量原始数据镜像。

成功 MP4 SHA-256：`238a8bbaf17af14c2bcf56bc35663364ddd348ee4275b49aa30bc297def72b7e`。

## 代码与运行依赖

[v63 冻结清单](diagnostics/full-visual-v63-hashes-20261017.json)覆盖 115 个文件；[导出来源](evidence/export-provenance.json)记录额外脚本、文档和证据的来源及 SHA。四个保护文件 `scene1/residual.py`、`scene1/settlement.py`、`scene1/wrist.py`、`config.yaml` 保持原字节。

本仓库 Release `v63-20261017` 的 `v63-runtime-assets-20261017.tar.gz` 保存原 ACT 权重和预处理/后处理文件、原 `go.bag` 准备动作、LeTools 部署配置及相关代码。它们是既有运行依赖的备份，未用于增加新的控制依据。依赖清单及归档 SHA 见 [runtime-assets.json](evidence/runtime-assets.json)。还需要原 ROS Noetic、Kuavo SDK、Python/CUDA 环境和官方仿真；此备份不包含整套 Docker 镜像、官方受保护场景资产或训练数据集。

已有环境中恢复到**独立项目目录**，避免覆盖活动研发项目：

1. 克隆仓库并执行 `python3 verify_backup.py`。
2. 从私有 Release 下载运行依赖归档，按 `evidence/runtime-assets.json` 核对 SHA。归档根相对于原 `/root/LeTools-Learning`；先解到新暂存目录核对，不直接覆盖正在工作的环境。
3. 需要运行时，将确认后的项目部署到算法容器 `/root/scene1_act_optimization`，恢复公共机器人 URDF 与腕标定。仅在无控制器、无活动实验时操作。
4. 按[当时完整交接文档](docs/HANDOFF-v63-20261017.md)第 7 节，通过官方入口完整复位场景；确认三路相机新鲜度与初始状态后启动：

```bash
docker exec kuavo-ros-noetic-s12-ready bash /root/scene1_act_optimization/run.sh \
  --config configs/visual_cycle_four_parcels.yaml --task go_run --mode optimized \
  --stop-after-parcels 4 --max-steps 16800
```

交接文档中的“当前状态”指 **v63 冻结时刻**，不是工作站后续研发状态；旧 ROS 检查点不能恢复到新场景。两份 HTML 复盘的旧首包统计链接属于历史资料，不改变上表五次 v63 结果。

## 已知限制

首包落稳失败尚未在 v63 修复。原候选高度被相邻点群污染是诊断假设，稀疏单帧回放不能证明连续落稳或唯一物理根因。现有版本未完成实际提速与多 seed 验证。正式成功录像的辅助 ROS 反馈有采集间隔；逐步控制日志完整、视频连续，不能称全量 ROS 反馈全程连续。

所有原抓取、释放、落稳、再抓、真实双边缘确认、插入限制、实际接触后交接和箱内视觉确认均保留。不得用本仓库的成功记录作为绕过检查或扩大运动边界的理由。
