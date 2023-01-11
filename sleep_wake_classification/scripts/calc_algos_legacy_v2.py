"""Use legacy algorithms from activity counts
"""
import pandas as pd
from utils import (
    cole,
    kripke,
    oakley,
    sadeh,
    sazonov,
    sazonov2,
    webster,
    webster_rescoring_rules,
    output_name,
)


def legacy_algos_v2(input_name, output_files, algo_names=["kripke"], col="zcounts"):

    counts = pd.read_csv(input_name)

    df_meta = counts[["timestamp", "subjid", "sleepstate"]]
    df_meta["is_sleep_all_sleep"] = 1
    df_meta["is_sleep_all_wake"] = 0

    counts = counts.rename(columns={col: "activity", "sleepstate": "psg"})

    for algo_name in algo_names:
        if "oakley" in algo_name:  # 'oakley10', 'oakley40', 'oakley80'
            theta = int(algo_name.replace("oakley", ""))
            pred_prob, pred = oakley(counts, theta)
        else:  # ['kripke', 'cole', 'sazonov', 'sazonov2', 'sadeh', 'webster']
            pred_prob, pred = eval(f"{algo_name}(counts)")
        df_meta[f"is_sleep_{algo_name}_palotti"] = pred
        df_meta[f"p_is_sleep_{algo_name}_palotti"] = pred_prob
        df_meta[f"is_sleep_{algo_name}_palotti_rsc"] = webster_rescoring_rules(pred)

    for k, gr in df_meta.groupby("subjid"):
        gr.drop("subjid", axis=1, inplace=True)
        gr.reset_index(drop=True, inplace=True)
        gr.to_csv(
            output_name(k, output_files),
            index=False,
        )


if __name__ == "__main__":
    algos = snakemake.params.algo_name.split(",")
    output_files = snakemake.output
    legacy_algos_palotti(snakemake.input.file_in, output_files, algos, snakemake.params.col)
