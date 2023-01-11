"""Function to combine stats per file from all algorithms into one csv file."""
import pandas as pd


def combine_perfile_results(input_files, fname_output):
    pd.concat([pd.read_csv(f) for f in input_files]).to_csv(fname_output)


if __name__ == "__main__":
    combine_perfile_results(snakemake.input,
                            snakemake.output[0])
