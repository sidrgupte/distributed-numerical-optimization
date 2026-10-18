
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

P = np.array([4, 8, 16])
N_values = [64, 256, 1024, 4096]

serial = {
    64: 0.000577,
    256: 0.033969,
    1024: 2.182096,
    4096: 138.313363,
}

parallel = {
    64: np.array([0.000173, 0.000143, 0.000219]),
    256: np.array([0.008598, 0.007210, 0.004639]),
    1024: np.array([0.545535, 0.461393, 0.358664]),
    4096: np.array([36.982388, 29.911572, 23.350082]),
}

speedup = {n: serial[n] / parallel[n] for n in N_values}
efficiency = {n: 100 * speedup[n] / P for n in N_values}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def finish(filename):
    plt.xticks(P)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / filename, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(7, 4.5))

for n in N_values:
    plt.plot(P, parallel[n], "o-", linewidth=2, label=f"N = {n}")

plt.yscale("log")
plt.xlabel("MPI processes (P)")
plt.ylabel("Parallel execution time (seconds)")
plt.title("Strong Scaling: 1D Distributed Matrix Multiplication")
plt.legend()
finish("execution_time.png")

plt.figure(figsize=(7, 4.5))

for n in N_values:
    plt.plot(P, speedup[n], "o-", linewidth=2, label=f"N = {n}")

plt.plot(P, P, "k--", linewidth=1.5, label="Ideal speedup")
plt.xlabel("MPI processes (P)")
plt.ylabel("Speedup")
plt.title("Strong Scaling Speedup")
plt.legend()
finish("speedup.png")

plt.figure(figsize=(7, 4.5))

for n in N_values:
    plt.plot(P, efficiency[n], "o-", linewidth=2, label=f"N = {n}")

plt.axhline(100, color="black", linestyle="--", linewidth=1.5, label="Ideal efficiency")
plt.xlabel("MPI processes (P)")
plt.ylabel("Parallel efficiency (%)")
plt.title("Parallel Efficiency")
plt.legend()
finish("efficiency.png")


phases = {
    64: {
        "Scatter": [0.000004, 0.000007, 0.000008],
        "Broadcast": [0.000006, 0.000014, 0.000062],
        "Compute": [0.000159, 0.000118, 0.000093],
        "Gather": [0.000027, 0.000065, 0.000147],
    },
    256: {
        "Scatter": [0.000019, 0.000022, 0.000031],
        "Broadcast": [0.000092, 0.000176, 0.000341],
        "Compute": [0.008454, 0.006951, 0.004295],
        "Gather": [0.000226, 0.002901, 0.000927],
    },
    1024: {
        "Scatter": [0.000942, 0.001054, 0.000371],
        "Broadcast": [0.002881, 0.003496, 0.007173],
        "Compute": [0.542824, 0.455882, 0.345048],
        "Gather": [0.017917, 0.155277, 0.087671],
    },
    4096: {
        "Scatter": [0.013350, 0.014042, 0.005961],
        "Broadcast": [0.054556, 0.046600, 0.120761],
        "Compute": [36.938583, 29.831162, 23.196501],
        "Gather": [1.591441, 10.659065, 4.948847],
    },
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

for ax, n in zip(axes.flat, N_values):
    for phase, times in phases[n].items():
        ax.plot(P, times, "o-", linewidth=1.8, label=phase)

    ax.set_title(f"N = {n}")
    ax.set_xticks(P)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("MPI processes (P)")
    ax.set_ylabel("Phase duration (seconds)")

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.suptitle(
    "MPI Phase Timings (Median Rank-Wise Maxima)",
    fontsize=15,
    y=0.99
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.955),
    ncol=4,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.89])

fig.savefig(
    out / "phase_timings.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print(f"Saved figures to {out.resolve()}")
