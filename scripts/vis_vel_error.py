#!/usr/bin/env python3
"""
可视化脚本：对比三种系统（RealFlight / Elastic-Tracker / visPlanner）
在三种轨迹（circle / typeD / type8）下的相对速度误差随时间变化，
以体现 RealFlight 的跟踪优越性。

速度均由位置序列中心差分自行计算（与 vis_traj.py 一致），
再经高斯平滑后求无人机与目标之间的相对速度误差范数。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# ──────────────────────────────────────────────
# 全局样式（与 vis_traj.py 保持一致）
# ──────────────────────────────────────────────
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
# 路径配置
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PARENT_DIR   = os.path.dirname(PROJECT_ROOT)

DATA_SOURCES = {
    'RealFlight':      os.path.join(PROJECT_ROOT,                    'test_log'),
    'Elastic-Tracker': os.path.join(PARENT_DIR, 'Elastic-Tracker',   'test_log'),
    'visPlanner':      os.path.join(PARENT_DIR, 'visPlanner',         'test_log'),
}

TRAJECTORY_TYPES = ['circle', 'typeD', 'type8']

OUT_PATH = os.path.join(PROJECT_ROOT, 'figs', 'vel_error_comparison.png')

# ──────────────────────────────────────────────
# 颜色配置
# ──────────────────────────────────────────────
SYSTEM_COLORS = {
    'RealFlight':      '#E53935',
    'Elastic-Tracker': '#1E88E5',
    'visPlanner':      '#43A047',
}

TRAJ_TITLES = {
    'circle': 'Circular Trajectory',
    'typeD':  'Type-D Trajectory',
    'type8':  'Figure-8 Trajectory',
}

GAUSS_SIGMA  = 3    # 速度计算后的高斯平滑 sigma（与 vis_traj.py 一致）
SMOOTH_SIGMA = 5    # 误差曲线展示平滑 sigma


# ──────────────────────────────────────────────
# 工具函数：中心差分计算水平速度（vis_traj.py 同款，仅 x-y 平面）
# ──────────────────────────────────────────────
def central_diff_velocity_2d(x: np.ndarray, y: np.ndarray,
                              ts: np.ndarray):
    """
    对位置序列 (x, y) 用中心差分求水平速度分量，
    首末点分别用前向/后向差分，结果经高斯平滑（sigma=GAUSS_SIGMA）。
    只使用水平分量可完全规避 z 轴传感器噪声与异常跳变的干扰。
    返回 vx, vy（均已平滑）。
    """
    n = len(x)
    vx = np.empty(n)
    vy = np.empty(n)

    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            vx[i] = (x[1] - x[0]) / dt
            vy[i] = (y[1] - y[0]) / dt
        elif i == n - 1:
            dt = ts[i] - ts[i - 1]
            vx[i] = (x[i] - x[i - 1]) / dt
            vy[i] = (y[i] - y[i - 1]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            vx[i] = (x[i + 1] - x[i - 1]) / dt
            vy[i] = (y[i + 1] - y[i - 1]) / dt

    vx = gaussian_filter1d(vx, sigma=GAUSS_SIGMA)
    vy = gaussian_filter1d(vy, sigma=GAUSS_SIGMA)
    return vx, vy


# ──────────────────────────────────────────────
# 数据加载与预处理
# ──────────────────────────────────────────────
def load_vel_error(log_dir: str, traj_type: str, system_name: str):
    """
    读取 CSV，从位置差分自行计算无人机和目标速度，
    裁剪目标静止段，归一化时间轴，返回 (t, vel_error) 或 None。
    """
    csv_file = os.path.join(log_dir, f'{traj_type}.csv')
    if not os.path.exists(csv_file):
        print(f"  [Warning] 文件不存在: {csv_file}")
        return None

    data = pd.read_csv(csv_file).dropna()
    data = data.sort_values('timestamp').reset_index(drop=True)

    if len(data) < 3:
        return None

    ts = data['timestamp'].values

    # ── 先用目标水平位置计算速度大小，用于确定截断点 ──
    tx = data['target_x'].values
    ty = data['target_y'].values
    tvx_raw, tvy_raw = central_diff_velocity_2d(tx, ty, ts)
    tgt_speed = np.sqrt(tvx_raw**2 + tvy_raw**2)

    # ── 截断：只保留目标运动阶段（与 vis_target_position.py 逻辑一致）──
    is_moving = tgt_speed > 0.05
    stop_idxs = [i for i in range(1, len(is_moving))
                 if is_moving[i - 1] and not is_moving[i]]

    if stop_idxs:
        cutoff = stop_idxs[0]
        if traj_type == 'type8':
            stop_ts = ts[cutoff]
            ext = cutoff
            for i in range(cutoff, len(data)):
                if ts[i] - stop_ts <= 0.50:
                    ext = i + 1
                else:
                    break
            cutoff = ext
        data = data.iloc[:cutoff].reset_index(drop=True)

    if len(data) < 3:
        return None

    ts = data['timestamp'].values

    # ── 从水平位置自行计算无人机速度 ─────────────
    dvx, dvy = central_diff_velocity_2d(
        data['drone_x'].values, data['drone_y'].values, ts)

    # ── 从水平位置自行计算目标速度 ────────────────
    tvx, tvy = central_diff_velocity_2d(
        data['target_x'].values, data['target_y'].values, ts)

    # ── 水平相对速度误差 ||v_drone_xy - v_target_xy|| ──
    vel_error = np.sqrt((dvx - tvx)**2 + (dvy - tvy)**2)

    t = ts - ts[0]   # 时间轴归零

    print(f"  [{system_name}] {traj_type}: {len(t)} 样本, "
          f"时长 {t[-1]:.1f}s, 均值误差 {np.mean(vel_error):.3f} m/s")
    return t, vel_error


# ──────────────────────────────────────────────
# 加载全部数据
# ──────────────────────────────────────────────
print("=== 加载所有轨迹数据 ===")
all_data: dict = {}

for sys_name, log_dir in DATA_SOURCES.items():
    all_data[sys_name] = {}
    print(f"\n[{sys_name}]")
    for traj in TRAJECTORY_TYPES:
        result = load_vel_error(log_dir, traj, sys_name)
        if result is not None:
            all_data[sys_name][traj] = result

# ──────────────────────────────────────────────
# 对齐时间轴：同一轨迹类型取所有系统的最短时长截断
# ──────────────────────────────────────────────
print("\n=== 对齐时间轴（取最短时长）===")
for traj in TRAJECTORY_TYPES:
    durations = {
        s: all_data[s][traj][0][-1]
        for s in all_data if traj in all_data[s]
    }
    if not durations:
        continue
    t_min = min(durations.values())
    print(f"  {traj}: " +
          "  ".join(f"{s}={v:.1f}s" for s, v in durations.items()) +
          f"  → 截断至 {t_min:.1f}s")
    for s in durations:
        t, err = all_data[s][traj]
        mask = t <= t_min
        all_data[s][traj] = (t[mask], err[mask])

# ──────────────────────────────────────────────
# 绘图：2×2 布局
#   [0,0] circle 时序    [0,1] typeD 时序
#   [1,0] type8  时序    [1,1] 均值误差柱状图
# ──────────────────────────────────────────────
print("\n=== 开始绘图 ===")

fig = plt.figure(figsize=(18, 12), facecolor='#FAFAFA')
fig.suptitle(
    "Horizontal Velocity Error Comparison  (XY Plane)\n"
    "RealFlight  vs.  Elastic-Tracker  vs.  visPlanner",
    fontsize=17, fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97,
                       top=0.91, bottom=0.08)

legend_handles = [
    Line2D([0], [0], color=SYSTEM_COLORS[s], lw=2.2, label=s)
    for s in ['RealFlight', 'Elastic-Tracker', 'visPlanner']
]

summary_mean: dict = {s: {} for s in DATA_SOURCES}
summary_std:  dict = {s: {} for s in DATA_SOURCES}


def plot_vel_error_timeseries(ax, traj_type: str):
    ax.set_facecolor('#FAFAFA')
    has_data = False
    t_max_all = 0.0

    for sys_name in ['RealFlight', 'Elastic-Tracker', 'visPlanner']:
        if traj_type not in all_data[sys_name]:
            continue

        t, err = all_data[sys_name][traj_type]
        color  = SYSTEM_COLORS[sys_name]
        t_max_all = max(t_max_all, t[-1])

        err_smooth = gaussian_filter1d(err, sigma=SMOOTH_SIGMA)

        mean_val = float(np.mean(err))
        std_val  = float(np.std(err))
        summary_mean[sys_name][traj_type] = mean_val
        summary_std[sys_name][traj_type]  = std_val

        # 原始（轻度平滑）淡色背景
        ax.plot(t, err_smooth, color=color, lw=0.7, alpha=0.25)
        # 主展示趋势线
        err_trend = gaussian_filter1d(err, sigma=SMOOTH_SIGMA * 3)
        ax.plot(t, err_trend, color=color, lw=2.2, alpha=0.92,
                label=f"{sys_name}  μ={mean_val:.3f} m/s")
        # 均值水平虚线
        ax.axhline(mean_val, color=color, lw=1.0, ls='--', alpha=0.55)

        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=13, color='gray')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Horizontal Velocity Error (m/s)')
    ax.set_title(TRAJ_TITLES.get(traj_type, traj_type), fontweight='bold')
    ax.set_xlim(0, t_max_all)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, ls=':', alpha=0.45)


ax00 = fig.add_subplot(gs[0, 0])
plot_vel_error_timeseries(ax00, 'circle')

ax01 = fig.add_subplot(gs[0, 1])
plot_vel_error_timeseries(ax01, 'typeD')

ax10 = fig.add_subplot(gs[1, 0])
plot_vel_error_timeseries(ax10, 'type8')

# ── [1,1] 分组柱状图：均值误差汇总 ───────────
ax11 = fig.add_subplot(gs[1, 1])
ax11.set_facecolor('#FAFAFA')

systems   = ['RealFlight', 'Elastic-Tracker', 'visPlanner']
n_traj    = len(TRAJECTORY_TYPES)
n_sys     = len(systems)
bar_width = 0.22
x_base    = np.arange(n_traj)

for i, sys_name in enumerate(systems):
    means  = [summary_mean[sys_name].get(tr, np.nan) for tr in TRAJECTORY_TYPES]
    stds   = [summary_std[sys_name].get(tr,  np.nan) for tr in TRAJECTORY_TYPES]
    offset = (i - (n_sys - 1) / 2) * bar_width

    bars = ax11.bar(
        x_base + offset, means,
        bar_width,
        yerr=stds, capsize=4,
        color=SYSTEM_COLORS[sys_name],
        alpha=0.82,
        label=sys_name,
        error_kw=dict(elinewidth=1.2, ecolor='#333333', alpha=0.7),
        zorder=3
    )

    for bar, val, std in zip(bars, means, stds):
        if not np.isnan(val):
            ax11.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (std if not np.isnan(std) else 0) + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=7.5, color=SYSTEM_COLORS[sys_name],
                fontweight='bold'
            )

ax11.set_xticks(x_base)
ax11.set_xticklabels([TRAJ_TITLES[tr] for tr in TRAJECTORY_TYPES], fontsize=9)
ax11.set_ylabel('Mean Horizontal Velocity Error (m/s)')
ax11.set_title('Mean Horizontal Velocity Error Summary\n(lower is better)', fontweight='bold')
ax11.set_ylim(bottom=0)
ax11.legend(handles=legend_handles, fontsize=9, loc='upper right')
ax11.grid(True, axis='y', ls=':', alpha=0.45)
ax11.set_axisbelow(True)

# ──────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches='tight')
print(f"\n[saved] {OUT_PATH}")

# 控制台统计摘要
print("\n======== 速度误差统计摘要（Mean ± Std，单位 m/s）========")
header = f"{'System':<20}" + "".join(f"  {tr:<14}" for tr in TRAJECTORY_TYPES)
print(header)
print("-" * len(header))
for sys_name in systems:
    row = f"{sys_name:<20}"
    for tr in TRAJECTORY_TYPES:
        m = summary_mean[sys_name].get(tr, float('nan'))
        s = summary_std[sys_name].get(tr,  float('nan'))
        row += f"  {m:.3f} ± {s:.3f}"
    print(row)
print("=" * len(header))

print("\n======== RealFlight 速度误差改善幅度 ========")
for tr in TRAJECTORY_TYPES:
    rf = summary_mean['RealFlight'].get(tr, np.nan)
    print(f"  [{TRAJ_TITLES[tr]}]")
    for other in ['Elastic-Tracker', 'visPlanner']:
        ov = summary_mean[other].get(tr, np.nan)
        if not np.isnan(rf) and not np.isnan(ov) and ov > 0:
            impv = (ov - rf) / ov * 100
            print(f"    vs {other:<18}: {impv:+.1f}%  ({ov:.3f} → {rf:.3f} m/s)")

plt.show()
