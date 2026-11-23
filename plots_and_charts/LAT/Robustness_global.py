import matplotlib.pyplot as plt

from pathlib import Path

from sweep_plot import plot_sweeps


# ============================================================
# Robustness vs configured radius (GCG, AutoDAN, Soft-Prompts)
# ============================================================

robustness_figure = plot_sweeps(
    panels=[
        ("GCG", "GCG"),
        ("AutoDAN", "AutoDAN"),
        ("Soft-Prompts", "Soft-Prompts"),
    ],
    ylabel="ASR (%)",
    figsize=(8, 11),
)


# ============================================================
# Save
# ============================================================

output_dir = Path(__file__).resolve().parent

pdf_path = output_dir / "robustness_sweeps.pdf"
png_path = output_dir / "robustness_sweeps.png"

robustness_figure.savefig(pdf_path, bbox_inches="tight")
robustness_figure.savefig(png_path, dpi=300, bbox_inches="tight")

print(
    "Saved robustness figure to:\n"
    f"  {pdf_path}\n"
    f"  {png_path}"
)

plt.show()
