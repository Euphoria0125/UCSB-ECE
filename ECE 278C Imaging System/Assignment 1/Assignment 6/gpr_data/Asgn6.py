"""
Assignment 6 — Multi-Frequency GPR Imaging
Run this script where gpr_data.mat is accessible.

Outputs (saved to ./assignment6_results/):
 - multi_frequency_image_sequence.mp4   (128-frame video)
 - multi_aperture_range_sequence.mp4    (200-frame video)
 - final_image_partA.png
 - final_image_partB.png
 - key frame images
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time
import sys


# --------------------------- Parameters ---------------------------
MATFILE = "gpr_data.mat"       # input mat file (must contain variables 'F', 'f', 'da')
OUTDIR = Path("assignment6_results")
OUTDIR.mkdir(parents=True, exist_ok=True)

eps_r = 6.0                    # relative permittivity (given)
c = 299792458.0                # speed of light (m/s)
v = c / np.sqrt(eps_r)        # propagation speed in medium

# imaging grid (you can tune these)
num_x = 200                    # lateral samples
num_z = 200                    # depth samples
z_max = 1.5                    # max depth in meters (tunable)
z_min = 0.02                   # min depth (avoid zero)

# video settings
fps = 8
dpi = 120

# --------------------------- Load data ---------------------------
print("Loading", MATFILE)
mat = sio.loadmat(MATFILE)
# Expect variables: F (128x200 complex), f (128,), da (200,)
if 'F' not in mat or 'f' not in mat or 'da' not in mat:
    print("ERROR: expected variables 'F', 'f', 'da' in the .mat file")
    sys.exit(1)

F_data = mat['F']    # shape (128, 200)
f_vec = mat['f'].squeeze()  # (128,)
da = mat['da'].squeeze()    # (200,)

print("Shapes: F:", F_data.shape, "f:", f_vec.shape, "da:", da.shape)
print(f"Frequency range: {f_vec.min():.3e} - {f_vec.max():.3e} Hz")
print(f"Antenna span: {da.min():.3f} m to {da.max():.3f} m")

# --------------------------- Imaging grid ---------------------------
x_min, x_max = da.min() - 0.2, da.max() + 0.2
x_img = np.linspace(x_min, x_max, num_x)
z_img = np.linspace(z_min, z_max, num_z)
Xg, Zg = np.meshgrid(x_img, z_img)   # shape (nz, nx)
img_shape = Xg.shape
print("Image grid:", img_shape, " (z,x)")

# Precompute broadcastable arrays
NumReceivers = da.size
Da_arr = da.reshape(1, 1, NumReceivers)   # (1,1,Nr)
Ximg_b = Xg[:, :, None]                   # (nz,nx,1)
Zimg_b = Zg[:, :, None]                   # (nz,nx,1)

# frequency-dependent quantities
freqs = f_vec
num_freqs = freqs.size
wavelengths = v / freqs
kvec = 2 * np.pi / wavelengths

# --------------------------- PART A: Frequency sweep ---------------------------
print("\nPART A: Multi-frequency backward propagation (128 frequencies)")
cumulative_A = np.zeros(img_shape, dtype=np.complex128)
subimages_A = []  # magnitude frames

start_t = time.time()
# We'll compute distances R (nz,nx,Nr) inside loop to avoid storing huge 4D arrays
for n in range(num_freqs):
    k = kvec[n]
    s = F_data[n, :].reshape(1, 1, NumReceivers)  # broadcastable
    R = np.sqrt((Ximg_b - Da_arr)**2 + (Zimg_b)**2)   # (nz,nx,Nr)
    # Phase-only round-trip kernel
    kernel = np.exp(-1j * k * 2.0 * R)
    subimage = np.sum(kernel * s, axis=2)   # (nz,nx)
    cumulative_A += subimage
    subimages_A.append(np.abs(cumulative_A.copy()))
    if (n+1) % 16 == 0 or (n==0) or (n==num_freqs-1):
        print(f"  Frequency {n+1}/{num_freqs} done")
end_t = time.time()
print("Part A done in {:.1f} s".format(end_t-start_t))

# Save final composite image for Part A
final_mag_A = np.abs(cumulative_A)
plt.figure(figsize=(6,6))
plt.imshow(final_mag_A, extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel("x (m)"); plt.ylabel("depth z (m)")
plt.title("Part A — Final composite (frequency integration)")
plt.savefig(OUTDIR / "final_image_partA.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Create short video (128 frames). Downsample frames if desired to shorten.
print("Creating Part A video (may require ffmpeg)...")
figA, axA = plt.subplots(figsize=(6,5))
vmaxA = np.max(subimages_A[-1])
imA = axA.imshow(subimages_A[0], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]],
                 aspect='auto', vmin=0, vmax=vmaxA)
axA.set_xlabel("x (m)"); axA.set_ylabel("depth z (m)")
axA.set_title("Part A: frequency cumulative frame 1")
plt.colorbar(imA, ax=axA, label='Magnitude')

def update_A(i):
    imA.set_data(subimages_A[i])
    axA.set_title(f"Part A: cumulative frequencies = {i+1}/{len(subimages_A)}")
    return [imA]

# Try to write mp4; if ffmpeg missing, user can comment writer block and use save as gif
writer = FFMpegWriter(fps=fps)
aniA = FuncAnimation(figA, update_A, frames=len(subimages_A), blit=False)
mp4A_path = OUTDIR / "multi_frequency_image_sequence.mp4"
try:
    aniA.save(mp4A_path, writer=writer, dpi=dpi)
    print("Saved Part A video to", mp4A_path)
except Exception as e:
    print("Failed to save MP4 (ffmpeg?). Exception:", e)
    # fallback: save a few key frames
    for kf in [0, 31, 63, 95, 127]:
        plt.figure(figsize=(6,5))
        plt.imshow(subimages_A[kf], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto', vmin=0, vmax=vmaxA)
        plt.colorbar(label='Magnitude')
        plt.title(f"Part A key frame {kf+1}")
        plt.savefig(OUTDIR / f"partA_frame_{kf+1:03d}.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    print("Saved key frames for Part A instead.")

plt.close(figA)

# --------------------------- PART A: Shallow-layer (0–0.20 m) flattened image ---------------------------
print("\nGenerating Part A shallow 0–0.20 m image...")

final_A = np.abs(cumulative_A)

# New depth limit
depth_limit_A = 0.20
crop_idx_A = np.where(z_img <= depth_limit_A)[0]

if len(crop_idx_A) == 0:
    print("WARNING: No depth samples <= 0.20 m for Part A!")
else:
    max_idx_A = crop_idx_A[-1]

    final_A_cropped = final_A[:max_idx_A+1, :]
    zA_cropped = z_img[:max_idx_A+1]

    plt.figure(figsize=(14, 2.2))   # long & thin like classmate's
    plt.imshow(
        final_A_cropped,
        extent=[x_img[0], x_img[-1], zA_cropped[-1], zA_cropped[0]],
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar(label='Magnitude')
    plt.xlabel("Cross-Range (m)")
    plt.ylabel("Depth (m)")
    plt.title("Frequency Effect Convergence (Part A Final, Depth 0–0.20 m)")

    out_fig_A = OUTDIR / "partA_shallow_0_20m.png"
    plt.savefig(out_fig_A, dpi=160, bbox_inches='tight')
    plt.close()

    print("Saved Part A 0–0.20 m shallow image to:", out_fig_A)


# --------------------------- PART B: Aperture sweep (range profiles) ---------------------------
print("\nPART B: Aperture synthesis (200 receivers)")
# For each receiver idx r:
#   s_freq = F_data[:, r] (128)
#   range_profile = sum_over_freqs( s_freq[f] * exp(-1j * k_f * 2 * R_r(x,z)) )
# Then cumulative sum over receivers for video frames.

range_profiles = []   # store cumulative images after each receiver addition
cumulative_B = np.zeros(img_shape, dtype=np.complex128)

# Precompute R for each receiver? We'll vectorize per receiver
# For memory reasons compute per receiver in loop
start_t = time.time()
for r in range(NumReceivers):
    x_r = da[r]
    # distance from this receiver to all image points
    Rr = np.sqrt((Xg - x_r)**2 + (Zg)**2)   # shape (nz, nx)
    # accumulate over frequencies
    profile_r = np.zeros(img_shape, dtype=np.complex128)
    s_freq = F_data[:, r]   # (128,)
    for nf in range(num_freqs):
        k = kvec[nf]
        profile_r += s_freq[nf] * np.exp(-1j * k * 2.0 * Rr)
    cumulative_B += profile_r
    range_profiles.append(np.abs(cumulative_B.copy()))
    if (r+1) % 20 == 0 or r==0 or r==NumReceivers-1:
        print(f"  Receiver {r+1}/{NumReceivers} done")
end_t = time.time()
print("Part B done in {:.1f} s".format(end_t-start_t))

# Save final composite image for Part B
final_mag_B = np.abs(cumulative_B)
plt.figure(figsize=(6,6))
plt.imshow(final_mag_B, extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel("x (m)"); plt.ylabel("depth z (m)")
plt.title("Part B — Final composite (aperture integration)")
plt.savefig(OUTDIR / "final_image_partB.png", dpi=dpi, bbox_inches='tight')
plt.close()

# Create Part B video (200 frames)
print("Creating Part B video (may require ffmpeg)...")
figB, axB = plt.subplots(figsize=(6,5))
vmaxB = np.max(range_profiles[-1])
imB = axB.imshow(range_profiles[0], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]],
                 aspect='auto', vmin=0, vmax=vmaxB)
axB.set_xlabel("x (m)"); axB.set_ylabel("depth z (m)")
axB.set_title("Part B: receivers combined = 1")

def update_B(i):
    imB.set_data(range_profiles[i])
    axB.set_title(f"Part B: receivers combined = {i+1}/{len(range_profiles)}")
    return [imB]

aniB = FuncAnimation(figB, update_B, frames=len(range_profiles), blit=False)
mp4B_path = OUTDIR / "multi_aperture_range_sequence.mp4"
try:
    aniB.save(mp4B_path, writer=writer, dpi=dpi)
    print("Saved Part B video to", mp4B_path)
except Exception as e:
    print("Failed to save MP4 (ffmpeg?). Exception:", e)
    # fallback: save key frames
    for kf in [0, 19, 49, 99, 199]:
        plt.figure(figsize=(6,5))
        plt.imshow(range_profiles[kf], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto', vmin=0, vmax=vmaxB)
        plt.colorbar(label='Magnitude')
        plt.title(f"Part B key frame {kf+1}")
        plt.savefig(OUTDIR / f"partB_frame_{kf+1:03d}.png", dpi=dpi, bbox_inches='tight')
        plt.close()
    print("Saved key frames for Part B instead.")

plt.close(figB)

# --------------------------- PART B: Shallow-layer (0–0.20 m) flattened image ---------------------------
print("\nGenerating Part B shallow 0–0.20 m image...")

final_B = np.abs(cumulative_B)

# New depth limit
depth_limit_B = 0.20
crop_idx_B = np.where(z_img <= depth_limit_B)[0]

if len(crop_idx_B) == 0:
    print("WARNING: No depth samples <= 0.20 m for Part B!")
else:
    max_idx_B = crop_idx_B[-1]

    final_B_cropped = final_B[:max_idx_B+1, :]
    zB_cropped = z_img[:max_idx_B+1]

    plt.figure(figsize=(14, 2.2))
    plt.imshow(
        final_B_cropped,
        extent=[x_img[0], x_img[-1], zB_cropped[-1], zB_cropped[0]],
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar(label='Magnitude')
    plt.xlabel("Cross-Range (m)")
    plt.ylabel("Depth (m)")
    plt.title("Aperture Effect Convergence (Part B Final, Depth 0–0.20 m)")

    out_fig_B = OUTDIR / "partB_shallow_0_20m.png"
    plt.savefig(out_fig_B, dpi=160, bbox_inches='tight')
    plt.close()

    print("Saved Part B 0–0.20 m shallow image to:", out_fig_B)


# --------------------------- Save a few additional key frames ---------------------------
print("Saving key frames (PNG)")
kfA = [0, 31, 63, 95, 127]
for k in kfA:
    plt.figure(figsize=(6,5))
    plt.imshow(subimages_A[k], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto', vmin=0, vmax=vmaxA)
    plt.colorbar(label='Magnitude')
    plt.title(f"Part A key frame {k+1}")
    plt.savefig(OUTDIR / f"partA_keyframe_{k+1:03d}.png", dpi=dpi, bbox_inches='tight')
    plt.close()

kfB = [0, 19, 49, 99, 199]
for k in kfB:
    plt.figure(figsize=(6,5))
    plt.imshow(range_profiles[k], extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], aspect='auto', vmin=0, vmax=vmaxB)
    plt.colorbar(label='Magnitude')
    plt.title(f"Part B key frame {k+1}")
    plt.savefig(OUTDIR / f"partB_keyframe_{k+1:03d}.png", dpi=dpi, bbox_inches='tight')
    plt.close()

print("All outputs saved under:", OUTDIR.resolve())
print("Done.")
