#!/usr/bin/env python3
"""
可视化脚本：展示神经网络对目标物体速度的预测能力
数据来源：circle_with_vel.csv（无人机跟踪圆周运动目标的飞行日志）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "../test_log/circle_with_vel.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "../figs/vel_predict_vis.png")

COLORS = {
    "gt":   "#2196F3",   # 真值 — 蓝色
    "pred": "#F44336",   # 预测 — 红色
    "err":  "#FF9800",   # 误差 — 橙色
    "bg":   "#FAFAFA",
}

# ──────────────────────────────────────────────
# 读取数据
# ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
t = df["timestamp"].values

gt_vx  = df["target_vx"].values
gt_vy  = df["target_vy"].values
gt_vz  = df["target_vz"].values
pr_vx  = df["predicted_target_vx"].values
pr_vy  = df["predicted_target_vy"].values
pr_vz  = df["predicted_target_vz"].values

gt_spd  = np.sqrt(gt_vx**2  + gt_vy**2  + gt_vz**2)
pr_spd  = np.sqrt(pr_vx**2  + pr_vy**2  + pr_vz**2)
vel_err = df["velocity_error"].values

# 各分量误差
err_vx = pr_vx - gt_vx
err_vy = pr_vy - gt_vy
err_vz = pr_vz - gt_vz

# 目标轨迹（2D）
tgt_x = df["target_x"].values
tgt_y = df["target_y"].values

# ──────────────────────────────────────────────
# 统计摘要
# ──────────────────────────────────────────────
def rmse(a, b): return np.sqrt(np.mean((a - b)**2))
def mae(a, b):  return np.mean(np.abs(a - b))
def corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return pearsonr(a, b)[0]

stats = {
    "vx": dict(rmse=rmse(pr_vx,gt_vx), mae=mae(pr_vx,gt_vx), r=corr(pr_vx,gt_vx)),
    "vy": dict(rmse=rmse(pr_vy,gt_vy), mae=mae(pr_vy,gt_vy), r=corr(pr_vy,gt_vy)),
    "vz": dict(rmse=rmse(pr_vz,gt_vz), mae=mae(pr_vz,gt_vz), r=corr(pr_vz,gt_vz)),
    "spd":dict(rmse=rmse(pr_spd,gt_spd),mae=mae(pr_spd,gt_spd),r=corr(pr_spd,gt_spd)),
}

# ──────────────────────────────────────────────
# 绘图布局：3 行 × 3 列
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(20, 15), facecolor=COLORS["bg"])
fig.suptitle(
    "Target Velocity Prediction Analysis  (circle_with_vel)",
    fontsize=18, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.42, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

# 图例句柄（全局复用）
legend_handles = [
    Line2D([0],[0], color=COLORS["gt"],   lw=2, label="Ground Truth"),
    Line2D([0],[0], color=COLORS["pred"], lw=2, linestyle="--", label="Predicted"),
]

# ── 辅助：绘制单分量时序对比 ──────────────────
def plot_component(ax, label, gt, pr, stat):
    ax.set_facecolor(COLORS["bg"])
    ax.plot(t, gt, color=COLORS["gt"],   lw=1.4, label="Ground Truth")
    ax.plot(t, pr, color=COLORS["pred"], lw=1.2, ls="--", alpha=0.9, label="Predicted")
    ax.fill_between(t, gt, pr, alpha=0.12, color=COLORS["err"])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel(f"{label}  (m/s)", fontsize=9)
    ax.set_title(
        f"Target {label}  |  RMSE={stat['rmse']:.3f}  MAE={stat['mae']:.3f}  r={stat['r']:.3f}",
        fontsize=10
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)

# ── [0,0] Vx 时序 ─────────────────────────────
ax_vx = fig.add_subplot(gs[0, 0])
plot_component(ax_vx, "Vx", gt_vx, pr_vx, stats["vx"])

# ── [0,1] Vy 时序 ─────────────────────────────
ax_vy = fig.add_subplot(gs[0, 1])
plot_component(ax_vy, "Vy", gt_vy, pr_vy, stats["vy"])

# ── [0,2] Vz 时序 ─────────────────────────────
ax_vz = fig.add_subplot(gs[0, 2])
plot_component(ax_vz, "Vz", gt_vz, pr_vz, stats["vz"])

# ── [1,0] 速度大小时序对比 ───────────────────
ax_spd = fig.add_subplot(gs[1, 0])
ax_spd.set_facecolor(COLORS["bg"])
ax_spd.plot(t, gt_spd,  color=COLORS["gt"],   lw=1.4, label="Ground Truth")
ax_spd.plot(t, pr_spd,  color=COLORS["pred"], lw=1.2, ls="--", alpha=0.9, label="Predicted")
ax_spd.fill_between(t, gt_spd, pr_spd, alpha=0.12, color=COLORS["err"])
ax_spd.set_xlabel("Time (s)", fontsize=9)
ax_spd.set_ylabel("Speed (m/s)", fontsize=9)
ax_spd.set_title(
    f"Speed Magnitude  |  RMSE={stats['spd']['rmse']:.3f}  r={stats['spd']['r']:.3f}",
    fontsize=10
)
ax_spd.legend(handles=legend_handles, fontsize=8)
ax_spd.grid(True, ls=":", alpha=0.5)

# ── [1,1] 速度误差时序 ───────────────────────
ax_err = fig.add_subplot(gs[1, 1])
ax_err.set_facecolor(COLORS["bg"])
ax_err.plot(t, vel_err, color=COLORS["err"], lw=1.2, label="Velocity Error ||dv||")
ax_err.axhline(np.mean(vel_err), color="gray", ls="--", lw=1.2,
               label=f"Mean = {np.mean(vel_err):.3f}")
ax_err.fill_between(t, 0, vel_err, alpha=0.18, color=COLORS["err"])
ax_err.set_xlabel("Time (s)", fontsize=9)
ax_err.set_ylabel("Error (m/s)", fontsize=9)
ax_err.set_title("Prediction Error  ||v_pred - v_gt||", fontsize=10)
ax_err.legend(fontsize=8)
ax_err.grid(True, ls=":", alpha=0.5)

# ── [1,2] 各分量误差分布直方图 ───────────────
ax_hist = fig.add_subplot(gs[1, 2])
ax_hist.set_facecolor(COLORS["bg"])
bins = 40
ax_hist.hist(err_vx, bins=bins, alpha=0.55, color="#2196F3", label=f"ΔVx  μ={err_vx.mean():.3f}")
ax_hist.hist(err_vy, bins=bins, alpha=0.55, color="#4CAF50", label=f"ΔVy  μ={err_vy.mean():.3f}")
ax_hist.hist(err_vz, bins=bins, alpha=0.55, color="#9C27B0", label=f"ΔVz  μ={err_vz.mean():.3f}")
ax_hist.axvline(0, color="black", lw=1.2, ls="--")
ax_hist.set_xlabel("Prediction Error (m/s)", fontsize=9)
ax_hist.set_ylabel("Count", fontsize=9)
ax_hist.set_title("Error Distribution per Component", fontsize=10)
ax_hist.legend(fontsize=8)
ax_hist.grid(True, ls=":", alpha=0.5)

# ── [2,0] Vx 散点：预测 vs 真值 ──────────────
def scatter_corr(ax, label, gt, pr, stat, color):
    ax.set_facecolor(COLORS["bg"])
    vmin = min(gt.min(), pr.min())
    vmax = max(gt.max(), pr.max())
    ax.scatter(gt, pr, c=color, s=4, alpha=0.35)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2, label="y = x")
    ax.set_xlabel(f"Ground Truth {label} (m/s)", fontsize=9)
    ax.set_ylabel(f"Predicted {label} (m/s)", fontsize=9)
    ax.set_title(f"Scatter Correlation  {label}  |  r = {stat['r']:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

ax_sc_vx = fig.add_subplot(gs[2, 0])
scatter_corr(ax_sc_vx, "Vx", gt_vx, pr_vx, stats["vx"], "#2196F3")

# ── [2,1] Vy 散点 ─────────────────────────────
ax_sc_vy = fig.add_subplot(gs[2, 1])
scatter_corr(ax_sc_vy, "Vy", gt_vy, pr_vy, stats["vy"], "#4CAF50")

# ── [2,2] 目标 2-D 轨迹 + 速度箭头（降采样）──
ax_traj = fig.add_subplot(gs[2, 2])
ax_traj.set_facecolor(COLORS["bg"])

# 轨迹线（用时间着色）
sc = ax_traj.scatter(tgt_x, tgt_y, c=t, cmap="viridis", s=2, alpha=0.6, zorder=2)
plt.colorbar(sc, ax=ax_traj, label="Time (s)", pad=0.02)

# 每 40 个点绘制一次速度箭头（真值 vs 预测）
step = max(1, len(t) // 40)
idx  = np.arange(0, len(t), step)
scale = 0.2  # 箭头缩放
ax_traj.quiver(tgt_x[idx], tgt_y[idx], gt_vx[idx]*scale, gt_vy[idx]*scale,
               color=COLORS["gt"],   alpha=0.8, scale=1, scale_units="xy",
               width=0.004, label="Ground Truth")
ax_traj.quiver(tgt_x[idx], tgt_y[idx], pr_vx[idx]*scale, pr_vy[idx]*scale,
               color=COLORS["pred"], alpha=0.7, scale=1, scale_units="xy",
               width=0.003, label="Predicted")

ax_traj.set_xlabel("X (m)", fontsize=9)
ax_traj.set_ylabel("Y (m)", fontsize=9)
ax_traj.set_title("Target 2-D Trajectory & Velocity Directions", fontsize=10)
ax_traj.legend(fontsize=8, loc="upper right")
ax_traj.set_aspect("equal")
ax_traj.grid(True, ls=":", alpha=0.5)

# ──────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"[saved] {OUT_PATH}")

# 控制台打印统计摘要
print("\n========== 速度预测统计摘要 ==========")
for k, v in stats.items():
    print(f"  {k:4s}  RMSE={v['rmse']:.4f}  MAE={v['mae']:.4f}  r={v['r']:.4f}")
print(f"  总时长: {t[-1]:.2f} s   采样点数: {len(t)}")
print("=======================================")

plt.show()
