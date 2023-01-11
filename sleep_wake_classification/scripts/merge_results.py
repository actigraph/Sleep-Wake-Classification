"""script to take in performance results on every subject on every algo and produce final results"""
import pandas as pd


def merge_results(output_fname_mean, output_fname_std, files_in):

    dfs = []
    for file in files_in:
        dfs.append(pd.read_csv(file))
    df = pd.concat(dfs, axis=0)
    drop_cols = ["TP", "TN", "FP", "FN"]
    df = df.drop(columns=drop_cols)
    df["col_name"] = df["col_name"].str.replace("is_sleep_", "")
    fix_cols = ["acc", "sens", "spec", "prec", "f1"]
    df_mean = df.groupby(by="col_name").mean()
    df_mean[fix_cols] = df_mean[fix_cols] * 100
    df_mean = df_mean.round(1)
    df_std = df.groupby(by="col_name").std()

    # output
    df_mean.to_csv(output_fname_mean)
    df_std.to_csv(output_fname_std)


if __name__ == "__main__":
    merge_results(snakemake.output.mean, snakemake.output.std, snakemake.input.files_in)
