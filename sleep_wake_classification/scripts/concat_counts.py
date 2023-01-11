import os.path as op

import pandas as pd
from actihealth.data.core import Reading
from utils import get_psg_downsampled


def concat_counts(in_files, output_name):
    cnts_all = []
    for file in in_files:
        reading = Reading.from_h5(file)

        cnts = reading.datasets["countsxyz"].data
        psg_results = get_psg_downsampled(reading, 30)

        cnts = cnts.join(psg_results, how="right")

        cnts["subjid"] = op.split(file)[1]
        cnts.reset_index(inplace=True)

        cnts_all.append(cnts)

    counts = pd.concat(cnts_all, axis=0)
    counts.to_csv(output_name, index=False)


if __name__ == "__main__":
    concat_counts(snakemake.input, snakemake.output[0])
