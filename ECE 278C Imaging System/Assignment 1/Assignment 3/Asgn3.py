import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter

class PhaseOnlyImaging:
    def __init__(self, lambda_val=1.0):
        self.lambda_val = lambda_val
        self.k = 2 * np.pi / lambda_val
        self.spacing = lambda_val / 4.0
        
        # Receiver array
        self.x_r = np.arange(-30 * lambda_val, 30 * lambda_val + self.spacing, self.spacing)
        self.y_r = -60 * lambda_val
        
        # Image region
        self.x_i = self.x_r
        self.y_i = self.x_r
        
        # Create output directory
        self.fig_dir = Path("assignment3_results")
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Number of receivers: {len(self.x_r)}")
        print(f"Image size: {len(self.y_i)} × {len(self.x_i)}")
        print(f"Spacing: λ/4 = {self.spacing:.4f}")

    def greens_function_2d(self, R):
        """2D Green's function"""
        R = np.maximum(R, 1e-12)
        return (1.0 / np.sqrt(1j * self.lambda_val * R)) * np.exp(1j * self.k * R)

    def simulate_received_signal(self, sources):
        """Generate received signal at receiver array from point sources"""
        s_total = np.zeros_like(self.x_r, dtype=np.complex128)
        for (x_s, y_s) in sources:
            R = np.sqrt((self.x_r - x_s) ** 2 + (self.y_r - y_s) ** 2)
            s_total += self.greens_function_2d(R)
        return s_total

    def phase_only_signal(self, s):
        """Part A: Keep only phase information of received signal"""
        phase = np.angle(s)
        # Set magnitude to constant (unit magnitude)
        return np.exp(1j * phase)

    def phase_only_kernel(self, dist):
        """Part B: Phase-only kernel for backward propagation"""
        # Keep only phase information of Green's function
        return np.exp(-1j * self.k * dist)

    def reconstruct_direct_superposition(self, s, method='full'):
        """Direct superposition with phase-only options"""
        reconstructed_image = np.zeros((len(self.y_i), len(self.x_i)), dtype=np.complex128)
        
        for j, y in enumerate(self.y_i):
            # Distance from each receiver to each image pixel
            dx = self.x_i[:, None] - self.x_r[None, :]  # (Nx, Nr)
            dy = y - self.y_r  # scalar
            dist = np.sqrt(dx**2 + dy**2)
            
            if method == 'full':
                # Full complex kernel
                kernel = (1.0 / np.sqrt(-1j * self.lambda_val * dist)) * np.exp(-1j * self.k * dist)
            elif method == 'phase_only_kernel':
                # Part B: Phase-only kernel
                kernel = self.phase_only_kernel(dist)
            else:
                # Standard kernel
                kernel = (1.0 / np.sqrt(-1j * self.lambda_val * dist)) * np.exp(-1j * self.k * dist)
            
            # Sum contributions from all receivers
            reconstructed_image[j, :] = kernel @ s
            
            if j % 20 == 0:
                print(f"    Progress: {j}/{len(self.y_i)}")
        
        return reconstructed_image

    def compute_fft_spectrum(self, image, out_size=512):
        """Compute 512x512 FFT spectrum"""
        pad_y_total = out_size - image.shape[0]
        pad_y_before = pad_y_total // 2
        pad_y_after = pad_y_total - pad_y_before
        
        pad_x_total = out_size - image.shape[1]
        pad_x_before = pad_x_total // 2
        pad_x_after = pad_x_total - pad_x_before
        
        image_padded = np.pad(image, ((pad_y_before, pad_y_after), 
                                    (pad_x_before, pad_x_after)), 
                            mode='constant')
        
        return np.fft.fftshift(np.fft.fft2(image_padded))

    def plot_magnitude_distribution(self, reconstructed_image, sources, title, filename):
        """Plot only the magnitude distribution of reconstructed image"""
        plt.figure(figsize=(10, 8))
        
        # Plot reconstructed image magnitude - using viridis colormap (same as before)
        im = plt.imshow(np.abs(reconstructed_image),
                       extent=[self.x_i.min(), self.x_i.max(), self.y_i.min(), self.y_i.max()],
                       cmap='viridis', origin='lower', aspect='auto')
        plt.title(f'{title}\nMagnitude Distribution', fontsize=14)
        plt.xlabel('x (λ)')
        plt.ylabel('y (λ)')
        
        # Mark source locations with red stars (same as before)
        for (x_s, y_s) in sources:
            plt.plot(x_s, y_s, 'r*', markersize=12, mew=2, label='Source Location')
        if len(sources) > 0:
            plt.legend(loc='upper right')
        
        plt.colorbar(im, label='Magnitude')
        plt.tight_layout()
        plt.savefig(self.fig_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

    def plot_fft_spectrum(self, fft_spectrum, title, filename):
        """Plot only the FFT spectrum"""
        plt.figure(figsize=(8, 6))
        
        # Plot FFT spectrum - using plasma colormap (same as before)
        im = plt.imshow(np.log10(np.abs(fft_spectrum)**2 + 1e-9),
                       cmap='plasma', origin='lower')
        plt.title(f'{title}\n512×512 FFT Spectrum', fontsize=14)
        plt.xlabel('Spatial Frequency kx')
        plt.ylabel('Spatial Frequency ky')
        plt.colorbar(im, label='log10(Power)')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

    def plot_combined_results(self, reconstructed_image, fft_spectrum, sources, title_prefix, filename):
        """Plot both magnitude distribution and FFT spectrum side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot reconstructed image magnitude - using viridis colormap
        ax1 = axes[0]
        im1 = ax1.imshow(np.abs(reconstructed_image),
                        extent=[self.x_i.min(), self.x_i.max(), self.y_i.min(), self.y_i.max()],
                        cmap='viridis', origin='lower', aspect='auto')
        ax1.set_title(f'{title_prefix}\nMagnitude Distribution', fontsize=12)
        ax1.set_xlabel('x (λ)')
        ax1.set_ylabel('y (λ)')
        
        # Mark source locations with red stars
        for (x_s, y_s) in sources:
            ax1.plot(x_s, y_s, 'r*', markersize=10, mew=2, label='Source Location')
        if len(sources) > 0:
            ax1.legend(loc='upper right')
        
        plt.colorbar(im1, ax=ax1, label='Magnitude')
        
        # Plot FFT spectrum - using plasma colormap
        ax2 = axes[1]
        im2 = ax2.imshow(np.log10(np.abs(fft_spectrum)**2 + 1e-9),
                        cmap='plasma', origin='lower')
        ax2.set_title(f'{title_prefix}\n512×512 FFT Spectrum', fontsize=12)
        ax2.set_xlabel('Spatial Frequency kx')
        ax2.set_ylabel('Spatial Frequency ky')
        plt.colorbar(im2, ax=ax2, label='log10(Power)')
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")

def run_phase_only_analysis():
    """Run complete phase-only imaging analysis"""
    # Initialize imaging system
    imaging_system = PhaseOnlyImaging()
    
    print("\n" + "="*70)
    print("PHASE-ONLY IMAGING ANALYSIS")
    print("="*70)
    
    # Define sources for both parts
    source_single = [(0, 0)]
    sources_multiple = [
        (0.0, 15 * imaging_system.lambda_val),
        (-12 * imaging_system.lambda_val, -9 * imaging_system.lambda_val),
        (0.0, -9 * imaging_system.lambda_val),
        (12 * imaging_system.lambda_val, -9 * imaging_system.lambda_val),
    ]
    
    # Generate original signals
    print("\nGenerating original signals...")
    s_A_original = imaging_system.simulate_received_signal(source_single)
    s_B_original = imaging_system.simulate_received_signal(sources_multiple)
    
    # Part A: Phase-only received signal
    print("\n" + "="*70)
    print("PART A: Phase-Only Received Signal")
    print("="*70)
    
    s_A_phase = imaging_system.phase_only_signal(s_A_original)
    s_B_phase = imaging_system.phase_only_signal(s_B_original)
    
    # Single source - Part A
    print("\nPart A - Single Source: Phase-only signal")
    img_A_phase_signal = imaging_system.reconstruct_direct_superposition(s_A_phase, method='full')
    fft_A_phase_signal = imaging_system.compute_fft_spectrum(img_A_phase_signal)
    
    # Generate separate magnitude distribution plot for Part A single source
    imaging_system.plot_magnitude_distribution(img_A_phase_signal, source_single,
                                             'Part A: Single Source - Phase-Only Signal',
                                             'partA_single_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_A_phase_signal,
                                   'Part A: Single Source - Phase-Only Signal',
                                   'partA_single_fft.png')
    
    # Generate combined plot
    imaging_system.plot_combined_results(img_A_phase_signal, fft_A_phase_signal, source_single,
                                       'Part A: Single Source - Phase-Only Signal',
                                       'partA_single_combined.png')
    
    # Multiple sources - Part A
    print("\nPart A - Multiple Sources: Phase-only signal")
    img_B_phase_signal = imaging_system.reconstruct_direct_superposition(s_B_phase, method='full')
    fft_B_phase_signal = imaging_system.compute_fft_spectrum(img_B_phase_signal)
    
    imaging_system.plot_magnitude_distribution(img_B_phase_signal, sources_multiple,
                                             'Part A: Multiple Sources - Phase-Only Signal',
                                             'partA_multiple_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_B_phase_signal,
                                   'Part A: Multiple Sources - Phase-Only Signal',
                                   'partA_multiple_fft.png')
    
    imaging_system.plot_combined_results(img_B_phase_signal, fft_B_phase_signal, sources_multiple,
                                       'Part A: Multiple Sources - Phase-Only Signal',
                                       'partA_multiple_combined.png')
    
    # Part B: Phase-only kernel
    print("\n" + "="*70)
    print("PART B: Phase-Only Kernel")
    print("="*70)
    
    # Single source - Part B (phase-only kernel with original signal)
    print("\nPart B - Single Source: Phase-only kernel")
    img_A_phase_kernel = imaging_system.reconstruct_direct_superposition(s_A_original, method='phase_only_kernel')
    fft_A_phase_kernel = imaging_system.compute_fft_spectrum(img_A_phase_kernel)
    
    imaging_system.plot_magnitude_distribution(img_A_phase_kernel, source_single,
                                             'Part B: Single Source - Phase-Only Kernel',
                                             'partB_single_kernel_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_A_phase_kernel,
                                   'Part B: Single Source - Phase-Only Kernel',
                                   'partB_single_kernel_fft.png')
    
    imaging_system.plot_combined_results(img_A_phase_kernel, fft_A_phase_kernel, source_single,
                                       'Part B: Single Source - Phase-Only Kernel',
                                       'partB_single_kernel_combined.png')
    
    # Multiple sources - Part B (phase-only kernel with original signal)
    print("\nPart B - Multiple Sources: Phase-only kernel")
    img_B_phase_kernel = imaging_system.reconstruct_direct_superposition(s_B_original, method='phase_only_kernel')
    fft_B_phase_kernel = imaging_system.compute_fft_spectrum(img_B_phase_kernel)
    
    imaging_system.plot_magnitude_distribution(img_B_phase_kernel, sources_multiple,
                                             'Part B: Multiple Sources - Phase-Only Kernel',
                                             'partB_multiple_kernel_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_B_phase_kernel,
                                   'Part B: Multiple Sources - Phase-Only Kernel',
                                   'partB_multiple_kernel_fft.png')
    
    imaging_system.plot_combined_results(img_B_phase_kernel, fft_B_phase_kernel, sources_multiple,
                                       'Part B: Multiple Sources - Phase-Only Kernel',
                                       'partB_multiple_kernel_combined.png')
    
    # Part B: Combined phase-only signal and kernel
    print("\n" + "="*70)
    print("PART B: Combined Phase-Only Signal and Kernel")
    print("="*70)
    
    # Single source - Combined phase-only
    print("\nPart B - Single Source: Phase-only signal + kernel")
    img_A_phase_both = imaging_system.reconstruct_direct_superposition(s_A_phase, method='phase_only_kernel')
    fft_A_phase_both = imaging_system.compute_fft_spectrum(img_A_phase_both)
    
    imaging_system.plot_magnitude_distribution(img_A_phase_both, source_single,
                                             'Part B: Single Source - Phase-Only Both',
                                             'partB_single_both_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_A_phase_both,
                                   'Part B: Single Source - Phase-Only Both',
                                   'partB_single_both_fft.png')
    
    imaging_system.plot_combined_results(img_A_phase_both, fft_A_phase_both, source_single,
                                       'Part B: Single Source - Phase-Only Both',
                                       'partB_single_both_combined.png')
    
    # Multiple sources - Combined phase-only
    print("\nPart B - Multiple Sources: Phase-only signal + kernel")
    img_B_phase_both = imaging_system.reconstruct_direct_superposition(s_B_phase, method='phase_only_kernel')
    fft_B_phase_both = imaging_system.compute_fft_spectrum(img_B_phase_both)
    
    imaging_system.plot_magnitude_distribution(img_B_phase_both, sources_multiple,
                                             'Part B: Multiple Sources - Phase-Only Both',
                                             'partB_multiple_both_magnitude.png')
    
    imaging_system.plot_fft_spectrum(fft_B_phase_both,
                                   'Part B: Multiple Sources - Phase-Only Both',
                                   'partB_multiple_both_fft.png')
    
    imaging_system.plot_combined_results(img_B_phase_both, fft_B_phase_both, sources_multiple,
                                       'Part B: Multiple Sources - Phase-Only Both',
                                       'partB_multiple_both_combined.png')
    
    print(f"\nAll phase-only imaging results saved to '{imaging_system.fig_dir}' directory")
    
    return {
        'single_phase_signal': img_A_phase_signal,
        'single_phase_kernel': img_A_phase_kernel,
        'single_phase_both': img_A_phase_both,
        'multiple_phase_signal': img_B_phase_signal,
        'multiple_phase_kernel': img_B_phase_kernel,
        'multiple_phase_both': img_B_phase_both
    }

# Run complete analysis
if __name__ == "__main__":
    results = run_phase_only_analysis()