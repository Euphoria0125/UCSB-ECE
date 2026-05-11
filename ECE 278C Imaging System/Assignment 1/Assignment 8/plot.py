import numpy as np
import matplotlib.pyplot as plt
files = [
("sweep_D_method1.txt", "D (units of λ0)", "Error vs D Method1"),
("sweep_B_method1.txt", "B=f0 (GHz)", "Error vs Bandwidth Method1"),
("sweep_r0_method1.txt", "r0 (m)", "Error vs r0 Method1"),
("sweep_theta_method1.txt", "theta0 (deg)", "Error vs theta0 Method1"),
("sweep_N_method1.txt", "N", "Error vs N Method1"),
("sweep_D_method2.txt", "D (units of λ0)", "Error vs D Method2"),
("sweep_B_method2.txt", "B=f0 (GHz)", "Error vs Bandwidth Method2"),
("sweep_r0_method2.txt", "r0 (m)", "Error vs r0 Method2"),
("sweep_theta_method2.txt", "theta0 (deg)", "Error vs theta0 Method2"),
("sweep_N_method2.txt", "N", "Error vs N Method2"),
]
for fname, xlabel, title in files:
    try:
        data = np.genfromtxt(f"assignment8_results/{fname}", comments="#", dtype=float, skip_header=2)
        x = data[:,0]
        err = data[:,1]
        plt.figure(figsize=(6,4))
        plt.plot(x, err, marker='o', linewidth=2)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel("Localization Error (m)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"assignment8_results/{title.replace(' ','_')}.png", dpi=200)
        plt.close()
        print(f"Generated: {title.replace(' ','_')}.png")
    except Exception as e:
        print(f"Error processing {fname}: {e}")
print("All plots generated!")