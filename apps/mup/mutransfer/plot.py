import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

ROOT_DIR = "apps/mup/mutransfer"

def read_jsonl_as_df(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(data)


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def main():
    sns.set_theme(style="whitegrid")
    parameterizations = [("sp", r"SP"), ("mup", r"$\mu$P")]
    seeds = [1]
    widths = [256, 512, 1024, 2048]
    lrs = [
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
        0.001953125,
        0.0009765625,
        0.00048828125,
        0.000244140625,
        0.0001220703125,
        0.00006103515625,
        0.00003051757812,
    ]

    color_helper = MplColorHelper("viridis", 0, len(widths) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    layers = 2

    for parameterization_idx, (parameterization, parameterization_str) in enumerate(
        parameterizations
    ):
        ax = axes[parameterization_idx]
        optimal_lrs, optimal_losses = [], []

        for width_idx, width in enumerate(widths):
            avg_final_losses, sem_losses, lrs_to_plot = [], [], []

            for lr in lrs:
                losses = []
                for seed in seeds:
                    job_name = (
                        f"width{width}_depth{layers}_seed{seed}_lr{lr:.20f}".rstrip("0")
                    )
                    ckpt_path = os.path.join(
                        ROOT_DIR, parameterization, "out", job_name, "metrics.jsonl"
                    )
                    
                    if os.path.exists(ckpt_path):
                        ckpt_df = read_jsonl_as_df(ckpt_path)
                        try:
                            losses.append(
                                ckpt_df["loss/out"].ewm(alpha=0.9).mean().values[-1]
                            )
                        except Exception:
                            print(f"Warning: Missing data for {job_name}")
                            continue

                if losses:
                    avg_final_losses.append(np.mean(losses))
                    sem_losses.append(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                    lrs_to_plot.append(lr)

            if lrs_to_plot:
                avg_final_losses = np.array(avg_final_losses)
                sem_losses = np.array(sem_losses)
                ax.plot(
                    lrs_to_plot,
                    avg_final_losses,
                    label=width,
                    marker="o",
                    color=color_helper.get_rgb(width_idx),
                )
                ax.fill_between(
                    lrs_to_plot,
                    avg_final_losses - sem_losses,
                    avg_final_losses + sem_losses,
                    color=color_helper.get_rgb(width_idx),
                    alpha=0.33,
                )

                optimum_idx = np.argmin(avg_final_losses)
                optimal_lrs.append(lrs_to_plot[optimum_idx])
                optimal_losses.append(avg_final_losses[optimum_idx])

        ax.plot(optimal_lrs, optimal_losses, color="red", linestyle="none", marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Learning rate")
        ax.set_title(parameterization_str)
        # ax.set_ylim(5, 6)

    axes[0].legend(title="Width")
    axes[0].set_ylabel("Train Loss on\nDCLM")
    axes[1].yaxis.set_ticklabels([])
    axes[1].tick_params(axis="y", length=0, width=0)

    plt.tight_layout()
    plt.savefig("mutransfer_lr_search.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
