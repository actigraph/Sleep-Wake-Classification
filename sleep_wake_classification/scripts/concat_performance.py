import pandas as pd

dfs = [pd.read_csv(i_file) for i_file in snakemake.input.files]
df = pd.concat(dfs, join="inner")
df = df.reset_index().drop(columns=["index"])
df = df.sort_values(by=["file", "col_name"])
df.to_csv(snakemake.output[0], index=False)
