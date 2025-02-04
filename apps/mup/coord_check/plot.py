#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
from typing import Dict, List, Tuple


def read_jsonl_as_df(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame()
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return pd.DataFrame(data)


class ColorHelper:
    def __init__(self, cmap_name: str, start_val: float, stop_val: float):
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val: float) -> tuple:
        return self.scalar_map.to_rgba(val)


def load_experiment_data(root_dir: str, included_dirs: List[str]) -> Dict:
    df_dict = {}
    for parameterization in os.listdir(root_dir):
        if parameterization not in included_dirs:
            continue

        df_dict[parameterization] = {}
        out_dir = os.path.join(root_dir, parameterization, "out")

        if not os.path.exists(out_dir):
            continue

        for job_name in os.listdir(out_dir):
            metrics_path = os.path.join(out_dir, job_name, "metrics.jsonl")
            df_dict[parameterization][job_name] = read_jsonl_as_df(metrics_path)

    return df_dict


def create_activation_plot(
    df_dict: Dict,
    parameterizations: List[str],
    layer_types: List[Tuple],
    widths: List[int],
    seeds: List[int],
    t_max: int,
    output_path: str,
):
    sns.set_theme(style="whitegrid")
    color_helper = ColorHelper("coolwarm", 0, t_max)

    n_cols = len(layer_types)
    n_rows = len(parameterizations)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    for parameterization_idx, parameterization in enumerate(parameterizations):
        print(f"Processing parameterization: {parameterization}")

        results_matrix = np.zeros((len(layer_types), t_max, len(widths), len(seeds)))

        for width_idx, width in enumerate(widths):
            for seed_idx, seed in enumerate(seeds):
                job_name = f"width{width}_depth2_seed{seed}"
                try:
                    ckpt_df = df_dict[parameterization][job_name]
                    if len(ckpt_df) != t_max or ckpt_df["global_step"].max() != t_max:
                        print(f"Warning: Empty or invalid data for {job_name}: len={len(ckpt_df)}")
                        continue

                    for layer_type_idx, (layer_type, _, _) in enumerate(layer_types):
                        
                        values = ckpt_df[layer_type].dropna().values[:t_max].flatten()

                        assert len(values) == t_max, "Invalid data shape for values"
                        
                        results_matrix[layer_type_idx, :, width_idx, seed_idx] = values

                except KeyError:
                    print(f"Warning: Missing data for {job_name}")
                    continue

        for layer_type_idx, (layer_type, layer_type_str, _) in enumerate(layer_types):
            ax = axes[parameterization_idx, layer_type_idx]

            for t in range(t_max):
                means = []
                stderrs = []

                for width_idx, width in enumerate(widths):
                    results = results_matrix[layer_type_idx, t, width_idx]
                    nnz_results = results[results != 0]

                    if len(nnz_results) > 0:
                        means.append(nnz_results.mean())
                        stderrs.append(
                            np.std(nnz_results, ddof=1) / np.sqrt(len(nnz_results))
                        )
                    else:
                        means.append(np.nan)
                        stderrs.append(np.nan)

                means = np.array(means)
                stderrs = np.array(stderrs)

                ax.plot(
                    widths,
                    means + 1e-3 * t,
                    label=f"{t+1}",
                    color=color_helper.get_rgb(t),
                    marker=".",
                )
                ax.fill_between(
                    widths,
                    means - stderrs,
                    means + stderrs,
                    color=color_helper.get_rgb(t),
                    alpha=0.5,
                )
            ax.set_title(layer_type_str)
            ax.set_xlabel("Width")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")

            if layer_type_idx == 0:
                ax.set_ylabel("np.abs(activation).mean()")
                ax.legend(loc="upper left", fontsize=8, title="Step")

    plt.suptitle("Coordinate Check")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    EXCLUDED_DIRS = ["plot.ipynb", "README.md", "deprecated", "mup"]
    LAYER_TYPES = [
        ("tok_embeddings_act_abs_mean", "Word Embedding", (1e-2, 1e-1)),
        ("attention_act_abs_mean", "Attention Output", (1e-2, 1e2)),
        ("feed_forward_act_abs_mean", "FFN Output", (1e-1, 1e2)),
        ("output_act_abs_mean", "Output Logits", (1e-3, 1e1)),
    ]
    PARAMETERIZATIONS = ["sp", "mupv1", "mupv2", "mupv3"]
    WIDTHS = [256, 512, 1024, 2048, 4096]
    SEEDS = [1]
    T_MAX = 10
    OUTPUT_PATH = "activation_analysis.png"
    ROOT_DIR = "apps/mup/coord_check"

    df_dict = load_experiment_data(ROOT_DIR, PARAMETERIZATIONS)

    create_activation_plot(
        df_dict=df_dict,
        parameterizations=PARAMETERIZATIONS,
        layer_types=LAYER_TYPES,
        widths=WIDTHS,
        seeds=SEEDS,
        t_max=T_MAX,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
