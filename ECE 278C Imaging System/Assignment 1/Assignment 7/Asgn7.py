"""
Assignment 7 – Short Baseline Sensing Systems (Python version, with video + parameter sweep)

功能：
1. 生成短基线双站系统下的两路步进频率数据 g1(n), g2(n)
2. 做双站 backward propagation，得到目标 profile
3. 存每一帧的累积图像，生成 N 帧收敛视频 (frequency convergence)
4. 参数扫描：改变 D、f0(=B)、r0、theta0，计算估计位置与真实位置的误差
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path

# -------------------- 全局固定参数 --------------------
c = 3e8                      # 传播速度 (m/s)
N = 128                      # 频率点数
A = 1.0                      # 目标散射强度

# 成像区域设置（你可以根据自己之前调好的范围修改）
Nx = 181
Nz = 181
x_min, x_max = -3.0, 3.0     # 横向视场 (m)
z_min, z_max = 1.0, 4.0      # 深度范围 (m)

# 输出目录
out_dir = Path("assignment7_results")
out_dir.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------
# 工具函数：给定 f0, D, r0_true, theta0_deg，完成一次仿真并（可选）返回所有频率累积帧
# ----------------------------------------------------------------------
def simulate_once(f0, D, r0_true, theta0_deg,
                  return_frames=False):
    """
    运行一次短基线仿真：
      - 生成 g1, g2
      - 回波成像
      - 返回成像幅度、网格坐标、真实目标坐标
      - 如果 return_frames=True，再返回每个频率累积的幅度帧列表（用于做视频）
    """
    lam0 = c / f0

    # --- 几何：Tx, Rx1, Rx2, target ---
    Tx = np.array([0.0, 0.0])
    Rx1 = np.array([-D, 0.0])
    Rx2 = np.array([ D, 0.0])

    theta0 = np.deg2rad(theta0_deg)
    target = np.array([r0_true * np.cos(theta0),
                       r0_true * np.sin(theta0)])

    r0_dist = np.linalg.norm(target - Tx)
    r1_dist = np.linalg.norm(target - Rx1)
    r2_dist = np.linalg.norm(target - Rx2)

    # --- 频率采样：fn = ((n + N/2)/N) * f0 ---
    n = np.arange(N)
    fn = ((n + N/2) / N) * f0

    # --- 生成 g1, g2 ---
    tau1 = (r0_dist + r1_dist) / c
    tau2 = (r0_dist + r2_dist) / c
    g1 = A * np.exp(1j * 2 * np.pi * fn * tau1)
    g2 = A * np.exp(1j * 2 * np.pi * fn * tau2)

    # --- 成像网格 ---
    x_grid = np.linspace(x_min, x_max, Nx)
    z_grid = np.linspace(z_min, z_max, Nz)
    X, Z = np.meshgrid(x_grid, z_grid)

    # 距离场
    Tx_x, Tx_z = Tx
    Rx1_x, Rx1_z = Rx1
    Rx2_x, Rx2_z = Rx2
    RT = np.sqrt((X - Tx_x)**2  + (Z - Tx_z)**2)
    R1 = np.sqrt((X - Rx1_x)**2 + (Z - Rx1_z)**2)
    R2 = np.sqrt((X - Rx2_x)**2 + (Z - Rx2_z)**2)

    # --- Backward propagation ---
    image = np.zeros_like(X, dtype=complex)
    frames = []  # 存每一帧累积图像

    for k in range(N):
        phase1 = np.exp(-1j * 2*np.pi * fn[k] * (RT + R1) / c)
        phase2 = np.exp(-1j * 2*np.pi * fn[k] * (RT + R2) / c)

        image += phase1 * np.conj(g1[k]) + phase2 * np.conj(g2[k])

        if return_frames:
            frames.append(np.abs(image.copy()))

    img_mag = np.abs(image)

    if return_frames:
        return img_mag, x_grid, z_grid, target, frames
    else:
        return img_mag, x_grid, z_grid, target

# ----------------------------------------------------------------------
# 1) 基础仿真：Option B (r0=2.5m, theta=20deg)，并生成 N 帧收敛视频
# ----------------------------------------------------------------------
# -------------------- User Input --------------------
print("========== Assignment 7 Short Baseline Imaging ==========")
r0_base = float(input("target distance r0 (m)： "))
theta0_base = float(input("target angel θ0 (deg)： "))

# -------------------- 默认参数设置 --------------------
f0_base = 3e9               # 3 GHz (baseline frequency)
lam0_base = c / f0_base     # wavelength around 0.1 m
D_base = 2.0 * lam0_base     # baseline D = 2 λ0 (可以固定，也可以通过扫描改变)

print("\nCurrent simulation parameters：")
print(f"r0 = {r0_base} m, theta0 = {theta0_base} deg")
print(f"f0 = {f0_base/1e9:.2f} GHz, λ0 = {lam0_base:.4f} m, D = {D_base:.4f} m\n")


print("Running base simulation for video & final image...")
img_mag, x_grid, z_grid, target, frames = simulate_once(
    f0_base, D_base, r0_base, theta0_base, return_frames=True
)

# ---------- 保存最终图像 ----------
plt.figure(figsize=(6,5))
plt.imshow(img_mag, extent=[x_min, x_max, z_max, z_min],
           aspect='auto', cmap='jet')
plt.colorbar(label="Magnitude")
plt.scatter(target[0], target[1], c='white', marker='x', s=60, label="True Target")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Short-Baseline Target Profile (Final Image)")
plt.legend(loc="upper right")
plt.savefig(out_dir / "final_target_profile.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved final_target_profile.png")

# ---------- 生成 N 帧频率收敛视频 ----------
print("Creating frequency-convergence video (N frames)...")
fig, ax = plt.subplots(figsize=(6,5))
vmax = np.max(frames[-1])
im = ax.imshow(frames[0], extent=[x_min, x_max, z_max, z_min],
               aspect='auto', cmap='jet', vmin=0, vmax=vmax)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
cb = plt.colorbar(im, ax=ax, label="Magnitude")

def update_frame(k):
    im.set_data(frames[k])
    ax.set_title(f"Frequency Convergence (1 to {k+1}/{N})")
    return [im]

ani = FuncAnimation(fig, update_frame, frames=len(frames), blit=False)
writer = FFMpegWriter(fps=8)
video_path = out_dir / "frequency_convergence_sequence.mp4"
try:
    ani.save(video_path, writer=writer, dpi=150)
    print("Saved video:", video_path)
except Exception as e:
    print("Failed to save mp4 (maybe no ffmpeg). Error:", e)

plt.close(fig)

# ----------------------------------------------------------------------
# 2) 参数扫描：改变 D, f0(=B), r0, theta0，计算定位误差
# ----------------------------------------------------------------------

def estimate_error(f0, D, r0_true, theta0_deg):
    """
    返回：(x_hat, z_hat, err_dist, target_coord)
    """
    img_mag, x_grid, z_grid, target = simulate_once(f0, D, r0_true, theta0_deg,
                                                   return_frames=False)
    # 找最大值坐标
    idx_max = np.unravel_index(np.argmax(img_mag), img_mag.shape)
    z_hat = z_grid[idx_max[0]]
    x_hat = x_grid[idx_max[1]]

    # 真实坐标
    theta0 = np.deg2rad(theta0_deg)
    x_true = r0_true * np.cos(theta0)
    z_true = r0_true * np.sin(theta0)

    err = np.sqrt((x_hat - x_true)**2 + (z_hat - z_true)**2)
    return x_hat, z_hat, err, (x_true, z_true)

# --------- (a) 扫 D ----------
print("\nParameter sweep: D (Tx-Rx separation)")
D_factors = [1.0, 2.0, 3.0, 4.0]   # D = k * λ0_base
D_list = [k * lam0_base for k in D_factors]

with open(out_dir / "sweep_D_results.txt", "w") as f:
    f.write("# Sweep over D (in units of λ0_base), f0=3GHz, r0=2.5m, theta0=20°\n")
    f.write("D_factor\tD(m)\terr(m)\n")
    for k, D_val in zip(D_factors, D_list):
        _, _, err, _ = estimate_error(f0_base, D_val, r0_base, theta0_base)
        line = f"{k:.1f}\t{D_val:.4f}\t{err:.4f}\n"
        print(line.strip())
        f.write(line)

# --------- (b) 扫 B (这里 B=f0，所以实际是扫中心频率) ----------
print("\nParameter sweep: B (implemented as different center frequency f0)")
f0_list = [2e9, 3e9, 4e9, 5e9]
with open(out_dir / "sweep_B_results.txt", "w") as f:
    f.write("# Sweep over f0 (thus B=f0), D=2λ0_base, r0=2.5m, theta0=20°\n")
    f.write("f0(GHz)\terr(m)\n")
    for f0_val in f0_list:
        try:
            print("Computing for f0 =", f0_val/1e9, "GHz ...")
            _, _, err, _ = estimate_error(f0_val, D_base, r0_base, theta0_base)
            f.write(f"{f0_val/1e9:.2f}\t{err:.4f}\n")
            print(" -> err =", err)
        except Exception as e:
            print("Error occurred for f0 =", f0_val/1e9, "GHz:", e)
