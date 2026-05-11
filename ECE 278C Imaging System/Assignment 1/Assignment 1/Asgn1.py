import numpy as np
import matplotlib.pyplot as plt

lam0 = 1.0        # wavelength λ0
dx = dy = lam0/4  # sample interval
N = 512           # FFT
L = 30*lam0       # aperture radius

x = np.arange(-N//2, N//2) * dx
y = np.arange(-N//2, N//2) * dy
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Generate point source wave field
def wavefield(x0, y0, lam):
    Rn = np.sqrt((X-x0)**2 + (Y-y0)**2)
    h = np.exp(1j * 2 * np.pi * Rn / lam) / np.sqrt(1j * lam * Rn + 1e-9)
    h[Rn == 0] = 0
    return h

# Draw Wave field
def plot_field(field, title):
    plt.figure()
    plt.imshow(np.abs(field), cmap='gray')
    plt.title(title)
    plt.colorbar(label="Amplitude")
    plt.show()

# Draw Spectrum
def plot_fft(field, title):
    spectrum = np.fft.fftshift(np.fft.fft2(field, (N, N)))
    magnitude = np.abs(spectrum)
    plt.figure()
    plt.imshow(np.log1p(magnitude), cmap='jet')
    plt.title(title)
    plt.colorbar(label="Log Amplitude")
    plt.show()

# Part A (1) Single Point
h1 = wavefield(0, 0, lam0)
plot_field(h1, "Single Point Source - Wavefield")
plot_fft(h1, "Single Point Source - Fourier Spectrum")

# Part A (2) Three Sources(Same lam0)
h2 = (wavefield(0, 15*lam0, lam0) +
      wavefield(-12*lam0, -9*lam0, lam0) +
      wavefield(12*lam0, -9*lam0, lam0))
plot_field(h2, "Three Sources (Same λ0) - Wavefield")
plot_fft(h2, "Three Sources (Same λ0) - Fourier Spectrum")

# Part A (3) Three Sources(Different lam0)
h3 = (wavefield(0, 15*lam0, lam0) +
      wavefield(-12*lam0, -9*lam0, lam0/2) +
      wavefield(12*lam0, -9*lam0, 2*lam0))
plot_field(h3, "Three Sources (Different λ) - Wavefield")
plot_fft(h3, "Three Sources (Different λ) - Fourier Spectrum")

# Part B (4) Phase-only
phase_only = np.exp(1j * np.angle(h3))
plot_field(phase_only, "Phase-only - Wavefield")
plot_fft(phase_only, "Phase-only - Fourier Spectrum")
