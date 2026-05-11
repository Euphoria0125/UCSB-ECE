import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
c = 3e8
A = 1.0
Nx = 181
Nz = 181
x_min, x_max = -3.0, 3.0
z_min, z_max = 1.0, 4.0
Nr = 181
Ntheta = 181
theta_min, theta_max = -1.4, 1.4
r_min, r_max = 1.0, 4.0
out_dir = Path("assignment8_results")
out_dir.mkdir(exist_ok=True, parents=True)

def generate_signals(f0, D, r0_true, theta0_deg, N):
    lam0 = c / f0
    Tx = np.array([0.0, 0.0])
    Rx1 = np.array([-D, 0.0])
    Rx2 = np.array([D, 0.0])
    theta0 = np.deg2rad(theta0_deg)
    # Corrected: x = r0 * sin(theta0), z = r0 * cos(theta0)
    target = np.array([r0_true * np.sin(theta0), r0_true * np.cos(theta0)])
    r0_dist = np.linalg.norm(target - Tx)
    r1_dist = np.linalg.norm(target - Rx1)
    r2_dist = np.linalg.norm(target - Rx2)
    n = np.arange(N)
    fn = ((n + N/2) / N) * f0
    tau1 = (r0_dist + r1_dist) / c
    tau2 = (r0_dist + r2_dist) / c
    g1 = A * np.exp(1j * 2 * np.pi * fn * tau1)
    g2 = A * np.exp(1j * 2 * np.pi * fn * tau2)
    return g1, g2, fn, target
def get_cartesian_grid():
    x_grid = np.linspace(x_min, x_max, Nx)
    z_grid = np.linspace(z_min, z_max, Nz)
    X, Z = np.meshgrid(x_grid, z_grid)
    return X, Z, x_grid, z_grid
def get_polar_grid():
    theta_grid = np.linspace(theta_min, theta_max, Ntheta)
    r_grid = np.linspace(r_min, r_max, Nr)
    Theta, Rr = np.meshgrid(theta_grid, r_grid)
    return Theta, Rr, theta_grid, r_grid
def simulate_method1(f0, D, r0_true, theta0_deg, N=128, bw_percent=0.1, return_frames=False):

    # ====== 1. Correct Bandwidth Modeling ======
    # BW = bw_percent * f0
    BW = bw_percent * f0
    f_min = f0 - BW / 2
    f_max = f0 + BW / 2

    # True frequency sweep
    fn = np.linspace(f_min, f_max, N)
    Tx = np.array([0.0, 0.0])
    Rx1 = np.array([-D, 0.0])
    Rx2 = np.array([ D, 0.0])

    theta = np.deg2rad(theta0_deg)
    target = np.array([
        r0_true * np.sin(theta),
        r0_true * np.cos(theta)
    ])

    rT = np.linalg.norm(target - Tx)
    r1 = np.linalg.norm(target - Rx1)
    r2 = np.linalg.norm(target - Rx2)

    g1 = np.exp(1j * 2 * np.pi * fn * (rT + r1) / c)
    g2 = np.exp(1j * 2 * np.pi * fn * (rT + r2) / c)

    X, Z, x_grid, z_grid = get_cartesian_grid()
    Mono1 = np.array([-D / 2, 0.0])
    Mono2 = np.array([ D / 2, 0.0])

    R1 = np.sqrt((X - Mono1[0])**2 + (Z - Mono1[1])**2)
    R2 = np.sqrt((X - Mono2[0])**2 + (Z - Mono2[1])**2)

    image1 = np.zeros_like(X, dtype=complex)
    image2 = np.zeros_like(X, dtype=complex)
    frames = []

    for k in range(N):
        phase1 = np.exp(1j * 2 * np.pi * fn[k] * 2 * R1 / c)
        phase2 = np.exp(1j * 2 * np.pi * fn[k] * 2 * R2 / c)

        image1 += phase1 * np.conj(g1[k])
        image2 += phase2 * np.conj(g2[k])

        if return_frames:
            mag = np.abs(image1) * np.abs(image2)
            frames.append(mag.copy())

    mag = np.abs(image1) * np.abs(image2)

    if return_frames:
        return mag, x_grid, z_grid, target, frames
    else:
        return mag, x_grid, z_grid, target

def simulate_method2(f0, D, r0_true, theta0_deg, N=128, bw_percent=0.1, return_frames=False):
    BW = bw_percent * f0
    f_min = f0 - BW / 2
    f_max = f0 + BW / 2
    fn = np.linspace(f_min, f_max, N)

    Tx = np.array([0.0, 0.0])
    Rx1 = np.array([-D, 0.0])
    Rx2 = np.array([ D, 0.0])

    theta = np.deg2rad(theta0_deg)
    target = np.array([
        r0_true * np.sin(theta),
        r0_true * np.cos(theta)
    ])

    rT = np.linalg.norm(target - Tx)
    r1 = np.linalg.norm(target - Rx1)
    r2 = np.linalg.norm(target - Rx2)

    g1 = np.exp(1j * 2 * np.pi * fn * (rT + r1) / c)
    g2 = np.exp(1j * 2 * np.pi * fn * (rT + r2) / c)

    Theta, Rr, theta_grid, r_grid = get_polar_grid()

    B = 2 * D
    image_range = np.zeros_like(Theta, dtype=complex)
    image_bearing = np.zeros_like(Theta, dtype=complex)

    g_prod = g1 * g2
    g_diff = g1 * np.conj(g2)

    frames = []

    for k in range(N):
        phase_range = np.exp(1j * 2 * np.pi * fn[k] * 4 * Rr / c)
        phase_bearing = np.exp(1j * 2 * np.pi * fn[k] * B * np.sin(Theta) / c)

        image_range += phase_range * np.conj(g_prod[k])
        image_bearing += phase_bearing * np.conj(g_diff[k])

        if return_frames:
            mag = np.abs(image_range) * np.abs(image_bearing)
            frames.append(mag.copy())

    mag = np.abs(image_range) * np.abs(image_bearing)

    # Extract 1D profiles (FFT results equivalent via DFT)
    range_profile = np.abs(image_range[:, 0])
    bearing_profile = np.abs(image_bearing[0, :])

    if return_frames:
        return mag, theta_grid, r_grid, target, frames, range_profile, bearing_profile
    else:
        return mag, theta_grid, r_grid, target, range_profile, bearing_profile

def estimate_error_method1(f0, D, r0_true, theta0_deg, N=128):
    mag, x_grid, z_grid, target = simulate_method1(f0, D, r0_true, theta0_deg, N)
    idx_max = np.unravel_index(np.argmax(mag), mag.shape)
    x_hat = x_grid[idx_max[1]]
    z_hat = z_grid[idx_max[0]]
    x_true = target[0]
    z_true = target[1]
    err = np.sqrt((x_hat - x_true)**2 + (z_hat - z_true)**2)
    return x_hat, z_hat, err, (x_true, z_true)
def estimate_error_method2(f0, D, r0_true, theta0_deg, N=128):
    mag, theta_grid, r_grid, target = simulate_method2(f0, D, r0_true, theta0_deg, N)
    idx_max = np.unravel_index(np.argmax(mag), mag.shape)
    r_hat = r_grid[idx_max[0]]
    theta_hat = theta_grid[idx_max[1]]
    x_hat = r_hat * np.sin(theta_hat)
    z_hat = r_hat * np.cos(theta_hat)
    x_true = target[0]
    z_true = target[1]
    err = np.sqrt((x_hat - x_true)**2 + (z_hat - z_true)**2)
    return x_hat, z_hat, err, (x_true, z_true)
print("========== Assignment 8 Short Baseline Approximation Techniques ==========")
r0_base = 2.5
theta0_base = 20
f0_base = 3e9
lam0_base = c / f0_base
D_base = 2.0 * lam0_base
N_base = 128
print("\nCurrent simulation parameters：")
print(f"r0 = {r0_base} m, theta0 = {theta0_base} deg")
print(f"f0 = {f0_base/1e9:.2f} GHz, λ0 = {lam0_base:.4f} m, D = {D_base:.4f} m\n")
print("Running base simulation for method1 video & final image...")
img_mag, x_grid, z_grid, target, frames = simulate_method1(f0_base, D_base, r0_base, theta0_base, N_base, return_frames=True)
plt.figure(figsize=(6,5))
plt.imshow(img_mag, extent=[x_min, x_max, z_max, z_min], aspect='auto', cmap='jet')
plt.colorbar(label="Magnitude")
plt.scatter(target[0], target[1], c='white', marker='x', s=60, label="True Target")
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Method 1: Monostatic Approximation (Final Image)")
plt.legend(loc="upper right")
plt.savefig(out_dir / "final_target_profile_method1.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved final_target_profile_method1.png")
fig, ax = plt.subplots(figsize=(6,5))
vmax = np.max(frames[-1])
im = ax.imshow(frames[0], extent=[x_min, x_max, z_max, z_min], aspect='auto', cmap='jet', vmin=0, vmax=vmax)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
cb = plt.colorbar(im, ax=ax, label="Magnitude")
def update_frame(k):
    im.set_data(frames[k])
    ax.set_title(f"Method 1: Frequency Convergence (1 to {k+1}/{N_base})")
    return [im]
ani = FuncAnimation(fig, update_frame, frames=len(frames), blit=False)
writer = FFMpegWriter(fps=8)
video_path = out_dir / "frequency_convergence_method1.mp4"
try:
    ani.save(video_path, writer=writer, dpi=150)
    print("Saved video:", video_path)
except Exception as e:
    print("Failed to save mp4 (maybe no ffmpeg). Error:", e)
plt.close(fig)
print("Running base simulation for method2 video & final image...")
# ======================== 在原代码最前面加入（只需加这一个函数）=======================
def compute_1d_profiles_clean(f0, D, r0_true, theta0_deg, N=128, bw_percent=0.1,
                              r_samples=2048, theta_samples=2048):
    """
    直接从原始序列 g1(n), g2(n) 计算干净地计算两个 1D 剖面
    返回：r_vec, range_profile, theta_vec, bearing_profile
    """
    BW = bw_percent * f0
    f_min = f0 - BW / 2
    f_max = f0 + BW / 2
    fn = np.linspace(f_min, f_max, N)

    # 真实目标位置
    theta_rad = np.deg2rad(theta0_deg)
    target = np.array([r0_true * np.sin(theta_rad), r0_true * np.cos(theta_rad)])

    Tx  = np.array([0.0, 0.0])
    Rx1 = np.array([-D, 0.0])
    Rx2 = np.array([ D, 0.0])

    rT  = np.linalg.norm(target - Tx)
    r1  = np.linalg.norm(target - Rx1)
    r2  = np.linalg.norm(target - Rx2)

    # 原始回波（归一化幅度）
    g1 = np.exp(1j * 2 * np.pi * fn * (rT + r1) / c)
    g2 = np.exp(1j * 2 * np.pi * fn * (rT + r2) / c)

    s_range   = g1 * g2                    # 用于距离向 → 对应 4R 时延
    s_bearing = g1 * np.conj(g2)           # 用于方位向 → 对应基线相位差

    # ==================== 1. 距离剖面 R(r) ====================
    r_vec = np.linspace(1.0, 5.0, r_samples)       # 多采样，保证尖峰清晰
    range_profile = np.zeros(r_samples, dtype=float)

    for i, r in enumerate(r_vec):
        # 匹配滤波：补偿 4r/c 的双程时延（因为 g1 g2 包含了两次往返）
        matched = np.exp(-1j * 2 * np.pi * fn * 4 * r / c)
        range_profile[i] = np.abs(np.sum(s_range * matched))

    range_profile /= np.max(range_profile)

    # ==================== 2. 方位剖面 q(θ) ====================
    theta_vec = np.linspace(-1.4, 1.4, theta_samples)
    bearing_profile = np.zeros(theta_samples, dtype=float)
    B = 2 * D                                            # 基线长度

    for i, th in enumerate(theta_vec):
        # 补偿基线引入的相位差 B·sinθ
        matched = np.exp(-1j * 2 * np.pi * fn * B * np.sin(th) / c)
        bearing_profile[i] = np.abs(np.sum(s_bearing * matched))

    bearing_profile /= np.max(bearing_profile)

    return r_vec, range_profile, theta_vec, bearing_profile

# ======================== 替换掉原来那两段画 1D 图的代码（直接粘贴覆盖）=======================

print("Computing clean 1D range and bearing profiles...")

r_vec, range_profile, theta_vec, bearing_profile = compute_1d_profiles_clean(
    f0_base, D_base, r0_base, theta0_base, N=N_base, bw_percent=0.1
)

true_theta_rad = np.deg2rad(theta0_base)

# —— 1D Range Profile ——
plt.figure(figsize=(7, 4.5))
plt.plot(r_vec, range_profile, 'C0', lw=2)
plt.axvline(r0_base, color='red', linestyle='--', label=f'True r = {r0_base} m')
plt.xlabel("Hypothetical range variable r (m)")
plt.ylabel("Normalized magnitude")
plt.title("1D Range Profile R(r) from g₁(n)·g₂(n)\n")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "1d_range_profile_.png", dpi=300)
plt.close()
print("Saved 1d_range_profile_.png")

# —— 1D Bearing Profile ——
plt.figure(figsize=(7, 4.5))
plt.plot(theta_vec, bearing_profile, 'C3', lw=2)
plt.axvline(true_theta_rad, color='red', linestyle='--', label=f'True θ = {theta0_base}°')
plt.xlabel("Bearing angle θ (rad)")
plt.ylabel("Normalized magnitude")
plt.title("1D Bearing Profile q(θ) from g₁(n)·g₂*(n)\n")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "1d_bearing_profile_C.png", dpi=300)
plt.close()
print("Saved 1d_bearing_profile_.png")
img_mag, theta_grid, r_grid, target, frames, range_profile, bearing_profile = simulate_method2(f0_base, D_base, r0_base, theta0_base, N_base, return_frames=True)
plt.figure(figsize=(6,5))
plt.imshow(img_mag, extent=[theta_min, theta_max, r_max, r_min], aspect='auto', cmap='jet')
plt.colorbar(label="Magnitude")
theta_true = np.deg2rad(theta0_base)
plt.scatter(theta_true, r0_base, c='white', marker='x', s=60, label="True Target")
plt.xlabel("theta (rad)")
plt.ylabel("r (m)")
plt.title("Method 2: Polar Approximation (Final Image)")
plt.legend(loc="upper right")
plt.savefig(out_dir / "final_target_profile_method2.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved final_target_profile_method2.png")
fig, ax = plt.subplots(figsize=(6,5))
vmax = np.max(frames[-1])
im = ax.imshow(frames[0], extent=[theta_min, theta_max, r_max, r_min], aspect='auto', cmap='jet', vmin=0, vmax=vmax)
ax.set_xlabel("theta (rad)")
ax.set_ylabel("r (m)")
cb = plt.colorbar(im, ax=ax, label="Magnitude")
def update_frame(k):
    im.set_data(frames[k])
    ax.set_title(f"Method 2: Frequency Convergence (1 to {k+1}/{N_base})")
    return [im]
ani = FuncAnimation(fig, update_frame, frames=len(frames), blit=False)
writer = FFMpegWriter(fps=8)
video_path = out_dir / "frequency_convergence_method2.mp4"
try:
    ani.save(video_path, writer=writer, dpi=150)
    print("Saved video:", video_path)
except Exception as e:
    print("Failed to save mp4 (maybe no ffmpeg). Error:", e)
plt.close(fig)

# Add 1D range profile plot
plt.figure(figsize=(6,5))
plt.plot(r_grid, range_profile / np.max(range_profile))
plt.scatter(r0_base, 1, c='red', marker='x', label="True Range")
plt.xlabel("r (m)")
plt.ylabel("Normalized Magnitude")
plt.title("1D Range Profile R(r) from g1 g2")
plt.legend()
plt.savefig(out_dir / "1d_range_profile.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved 1d_range_profile.png")

# Add 1D bearing profile plot
plt.figure(figsize=(6,5))
plt.plot(theta_grid, bearing_profile / np.max(bearing_profile))
plt.scatter(theta_true, 1, c='red', marker='x', label="True Bearing")
plt.xlabel("theta (rad)")
plt.ylabel("Normalized Magnitude")
plt.title("1D Bearing Profile q(θ) from g1 g2*")
plt.legend()
plt.savefig(out_dir / "1d_bearing_profile.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved 1d_bearing_profile.png")

for method in ['method1', 'method2']:
    estimate_func = estimate_error_method1 if method == 'method1' else estimate_error_method2

    print(f"\n===== Improved Sweep for {method}: D =====")
    D_factors = np.linspace(1, 6, 15)
    with open(out_dir / f"sweep_D_{method}.txt", "w") as f:
        f.write("D_factor\tD(m)\terr(m)\n")
        for k in D_factors:
            D_val = k * lam0_base
            _, _, err, _ = estimate_func(f0_base, D_val, r0_base, theta0_base, N_base)
            f.write(f"{k:.3f}\t{D_val:.4f}\t{err:.4f}\n")

    print(f"\n===== Improved Sweep for {method}: Bandwidth =====")
    f0_list = np.linspace(1e9, 6e9, 20)
    with open(out_dir / f"sweep_B_{method}.txt", "w") as f:
        f.write("f0(GHz)\terr(m)\n")
        for f0_val in f0_list:
            _, _, err, _ = estimate_func(f0_val, D_base, r0_base, theta0_base, N_base)
            f.write(f"{f0_val/1e9:.3f}\t{err:.4f}\n")

    print(f"\n===== Improved Sweep for {method}: r0 =====")
    r0_list = np.linspace(1.5, 4.0, 20)
    with open(out_dir / f"sweep_r0_{method}.txt", "w") as f:
        f.write("r0(m)\terr(m)\n")
        for r0_val in r0_list:
            _, _, err, _ = estimate_func(f0_base, D_base, r0_val, theta0_base, N_base)
            f.write(f"{r0_val:.3f}\t{err:.4f}\n")

    print(f"\n===== Improved Sweep for {method}: theta0 =====")
    theta0_list = np.linspace(5, 175, 25)  # 25 angles, smooth U-shaped curve
    with open(out_dir / f"sweep_theta_{method}.txt", "w") as f:
        f.write("theta(deg)\terr(m)\n")
        for theta_val in theta0_list:
            _, _, err, _ = estimate_func(f0_base, D_base, r0_base, theta_val, N_base)
            f.write(f"{theta_val:.2f}\t{err:.4f}\n")

    print(f"\n===== Improved Sweep for {method}: N =====")
    N_list = np.linspace(16, 256, 16, dtype=int)
    with open(out_dir / f"sweep_N_{method}.txt", "w") as f:
        f.write("N\terr(m)\n")
        for n_val in N_list:
            _, _, err, _ = estimate_func(f0_base, D_base, r0_base, theta0_base, n_val)
            f.write(f"{n_val}\t{err:.4f}\n")

print("\nAll parameter sweeps completed!")
print("Results saved to txt files in assignment8_results.")

D_lambda_vals = np.linspace(5, 30, 6)
err1_D, err2_D, res1_D, res2_D = [], [], [], []
for d in D_lambda_vals:
    D_val = d * lam0_base
    mag1, _, _, target = simulate_method1(f0_base, D_val, r0_base, theta0_base, N_base)
    idx1 = np.unravel_index(np.argmax(mag1), mag1.shape)
    est1 = np.array(x_grid[idx1[1]], z_grid[idx1[0]])
    err1 = np.sqrt(np.sum((est1 - target)**2))
    profile1 = mag1[idx1[0], :]
    peak1 = np.max(profile1)
    res1 = np.sum(profile1 > peak1/2) * (x_grid[1] - x_grid[0])
    err1_D.append(err1); res1_D.append(res1)

    mag2, theta_grid, r_grid, _ = simulate_method2(f0_base, D_val, r0_base, theta0_base, N_base)
    idx2 = np.unravel_index(np.argmax(mag2), mag2.shape)
    r_hat = r_grid[idx2[0]]
    theta_hat = theta_grid[idx2[1]]
    est2 = np.array([r_hat * np.sin(theta_hat), r_hat * np.cos(theta_hat)])
    err2 = np.sqrt(np.sum((est2 - target)**2))
    profile2 = mag2[idx2[0], :]
    peak2 = np.max(profile2)
    res2 = np.sum(profile2 > peak2/2) * (theta_grid[1] - theta_grid[0]) * r0_base  # approximate angular resolution
    err2_D.append(err2); res2_D.append(res2)

bw_frac = np.linspace(0, 0.5, 6)
err1_B, err2_B, res1_B, res2_B = [], [], [], []
for b in bw_frac:
    f0_val = f0_base * (1 + b)
    mag1, _, _, target = simulate_method1(f0_val, D_base, r0_base, theta0_base, N_base)
    idx1 = np.unravel_index(np.argmax(mag1), mag1.shape)
    est1 = np.array(x_grid[idx1[1]], z_grid[idx1[0]])
    err1 = np.sqrt(np.sum((est1 - target)**2))
    profile1 = mag1[:, idx1[1]]
    peak1 = np.max(profile1)
    res1 = np.sum(profile1 > peak1/2) * (z_grid[1] - z_grid[0])
    err1_B.append(err1); res1_B.append(res1)

    mag2, theta_grid, r_grid, _ = simulate_method2(f0_val, D_base, r0_base, theta0_base, N_base)
    idx2 = np.unravel_index(np.argmax(mag2), mag2.shape)
    r_hat = r_grid[idx2[0]]
    theta_hat = theta_grid[idx2[1]]
    est2 = np.array([r_hat * np.sin(theta_hat), r_hat * np.cos(theta_hat)])
    err2 = np.sqrt(np.sum((est2 - target)**2))
    profile2 = mag2[:, idx2[1]]
    peak2 = np.max(profile2)
    res2 = np.sum(profile2 > peak2/2) * (r_grid[1] - r_grid[0])
    err2_B.append(err2); res2_B.append(res2)

N_vals = np.linspace(50, 300, 6, dtype=int)
err1_N, err2_N, res1_N, res2_N = [], [], [], []
for n in N_vals:
    mag1, _, _, target = simulate_method1(f0_base, D_base, r0_base, theta0_base, n)
    idx1 = np.unravel_index(np.argmax(mag1), mag1.shape)
    est1 = np.array(x_grid[idx1[1]], z_grid[idx1[0]])
    err1 = np.sqrt(np.sum((est1 - target)**2))
    profile1 = mag1[idx1[0], :]
    peak1 = np.max(profile1)
    res1 = np.sum(profile1 > peak1/2) * (x_grid[1] - x_grid[0])
    err1_N.append(err1); res1_N.append(res1)

    mag2, theta_grid, r_grid, _ = simulate_method2(f0_base, D_base, r0_base, theta0_base, n)
    idx2 = np.unravel_index(np.argmax(mag2), mag2.shape)
    r_hat = r_grid[idx2[0]]
    theta_hat = theta_grid[idx2[1]]
    est2 = np.array([r_hat * np.sin(theta_hat), r_hat * np.cos(theta_hat)])
    err2 = np.sqrt(np.sum((est2 - target)**2))
    profile2 = mag2[idx2[0], :]
    peak2 = np.max(profile2)
    res2 = np.sum(profile2 > peak2/2) * (theta_grid[1] - theta_grid[0]) * r0_base
    err2_N.append(err2); res2_N.append(res2)

theta_vals = np.linspace(20, 160, 8)
err1_theta, err2_theta, res1_theta, res2_theta = [], [], [], []
for th in theta_vals:
    mag1, _, _, target = simulate_method1(f0_base, D_base, r0_base, th, N_base)
    idx1 = np.unravel_index(np.argmax(mag1), mag1.shape)
    est1 = np.array(x_grid[idx1[1]], z_grid[idx1[0]])
    err1 = np.sqrt(np.sum((est1 - target)**2))
    profile1 = mag1[idx1[0], :]
    peak1 = np.max(profile1)
    res1 = np.sum(profile1 > peak1/2) * (x_grid[1] - x_grid[0])
    err1_theta.append(err1); res1_theta.append(res1)

    mag2, theta_grid, r_grid, _ = simulate_method2(f0_base, D_base, r0_base, th, N_base)
    idx2 = np.unravel_index(np.argmax(mag2), mag2.shape)
    r_hat = r_grid[idx2[0]]
    theta_hat = theta_grid[idx2[1]]
    est2 = np.array([r_hat * np.sin(theta_hat), r_hat * np.cos(theta_hat)])
    err2 = np.sqrt(np.sum((est2 - target)**2))
    profile2 = mag2[idx2[0], :]
    peak2 = np.max(profile2)
    res2 = np.sum(profile2 > peak2/2) * (theta_grid[1] - theta_grid[0]) * r0_base
    err2_theta.append(err2); res2_theta.append(res2)

fig, axs = plt.subplots(2,2, figsize=(12,9))
fig.suptitle("Part (d) - Accuracy Analysis", fontsize=14)
axs[0,0].plot(D_lambda_vals, err1_D, 'o-b', label="Method 1 (Mono)")
axs[0,0].plot(D_lambda_vals, err2_D, 's-r', label="Method 2 (Polar)")
axs[0,0].set_xlabel("Separation D (lambda)"); axs[0,0].set_ylabel("Error (m)"); axs[0,0].legend(); axs[0,0].grid(alpha=0.3)
axs[0,1].plot(bw_frac, err1_B, 'o-b'); axs[0,1].plot(bw_frac, err2_B, 's-r')
axs[0,1].set_xlabel("Bandwidth (% f0)"); axs[0,1].grid(alpha=0.3)
axs[1,0].plot(N_vals, err1_N, 'o-b'); axs[1,0].plot(N_vals, err2_N, 's-r')
axs[1,0].set_xlabel("Steps N"); axs[1,0].grid(alpha=0.3)
axs[1,1].plot(theta_vals, err1_theta, 'o-b'); axs[1,1].plot(theta_vals, err2_theta, 's-r')
axs[1,1].set_xlabel("Bearing (deg)"); axs[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir/"Part_d_Accuracy_Analysis.png", dpi=300); plt.close()

fig, axs = plt.subplots(2,2, figsize=(12,9))
fig.suptitle("Part (d) - Resolution Analysis", fontsize=14)
axs[0,0].plot(D_lambda_vals, res1_D, 'o-b', label="Method 1 (Mono)")
axs[0,0].plot(D_lambda_vals, res2_D, 's-r', label="Method 2 (Polar)")
axs[0,0].set_xlabel("Separation D (lambda)"); axs[0,0].set_ylabel("Resolution Cell Size (m)"); axs[0,0].legend(); axs[0,0].grid(alpha=0.3)
axs[0,1].plot(bw_frac, res1_B, 'o-b'); axs[0,1].plot(bw_frac, res2_B, 's-r')
axs[0,1].set_xlabel("Bandwidth (% f0)"); axs[0,1].grid(alpha=0.3)
axs[1,0].plot(N_vals, res1_N, 'o-b'); axs[1,0].plot(N_vals, res2_N, 's-r')
axs[1,0].set_xlabel("Steps N"); axs[1,0].grid(alpha=0.3)
axs[1,1].plot(theta_vals, res1_theta, 'o-b'); axs[1,1].plot(theta_vals, res2_theta, 's-r')
axs[1,1].set_xlabel("Bearing (deg)"); axs[1,0].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir/"Part_d_Resolution_Analysis.png", dpi=300); plt.close()