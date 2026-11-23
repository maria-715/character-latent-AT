import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from sweep_plot import plot_sweeps


# ============================================================
# Task capability vs configured radius
# ============================================================

capability_figure = plot_sweeps(
    panels=[
        ("MMLU", "MMLU"),
        ("ARC-E", "ARC-Easy"),
        ("ARC-C", "ARC-Challenge"),
    ],
    ylabel="Score (%)",
    figsize=(8, 11),
)


# ============================================================
# Benign behavioural utility vs configured radius
# ============================================================

safety_figure = plot_sweeps(
    panels=[
        ("Harmless", "Harmless"),
        ("XSTest", "XSTest"),
    ],
    ylabel="Score (%)",
    figsize=(8, 8.5),
    ymax_cap=105,
)


# ============================================================
# Save
# ============================================================

script_path = Path(__file__).resolve()
output_dir = script_path.parent

outputs = {
    "task_capability_sweeps": capability_figure,
    "safety_sweeps": safety_figure,
}

for stem, fig in outputs.items():
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")

combined_pdf_path = output_dir / f"{script_path.stem}.pdf"

with PdfPages(combined_pdf_path) as pdf:
    for fig in outputs.values():
        pdf.savefig(fig, bbox_inches="tight")

print(
    "Saved utility figures to:\n"
    + "".join(f"  {output_dir / stem}.pdf/.png\n" for stem in outputs)
    + f"\nSaved combined PDF to:\n  {combined_pdf_path}"
)

plt.show()
