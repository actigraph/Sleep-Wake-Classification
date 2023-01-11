import os.path as op

from actihealth.algorithms.sleep import RandomForestSleep2021Pipeline
from actihealth.reader.raw import RawNewcastlePSGReader
from utils import get_psg_downsampled


def sleep_classify_RF(file, output_fname, rf_model):
    reading = RawNewcastlePSGReader().read(file)
    psg_results = get_psg_downsampled(reading, 30)
    rf_model.apply(reading)

    rf_results = reading.datasets["random_forest_results"].data
    rf_results = rf_results.join(psg_results, how="right")
    rf_results = rf_results.rename(columns={"is_sleep": "is_sleep_RF"})
    assert rf_results.shape[0] == psg_results.size

    rf_results.to_csv(op.join(output_fname))


def main():
    rf = RandomForestSleep2021Pipeline(resample_origin="start")
    for in_file, out_file in zip(snakemake.input, snakemake.output):
        sleep_classify_RF(in_file, out_file, rf)


if __name__ == "__main__":
    main()
