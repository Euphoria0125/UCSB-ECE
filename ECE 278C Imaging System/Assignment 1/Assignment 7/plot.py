import numpy as np
import matplotlib.pyplot as plt

files = [
    ("sweep_D_results.txt", "D (units of λ0)", "Error vs D"),
    ("sweep_B_results.txt", "B=f0 (GHz)", "Error vs Bandwidth"),
    ("sweep_r0_results.txt", "r0 (m)", "Error vs r0"),
    ("sweep_theta_results.txt", "theta0 (deg)", "Error vs theta0"),
]

for fname, xlabel, title in files:
    try:
        data = np.genfromtxt(f"assignment7_results/{fname}", comments="#", dtype=float, skip_header=2)
        
        x = data[:,0]
        err = data[:,1]

        plt.figure(figsize=(6,4))
        plt.plot(x, err, marker='o', linewidth=2)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel("Localization Error (m)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"assignment7_results/{title.replace(' ','_')}.png", dpi=200)
        plt.close()
        print(f"Generated: {title.replace(' ','_')}.png")
    except Exception as e:
        print(f"Error processing {fname}: {e}")

print("All plots generated!")