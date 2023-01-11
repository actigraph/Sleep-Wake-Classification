import pandas as pd
from actihealth.reader.raw import RawNewcastlePSGReader
from actihealth.transform.counts import CountsXYZ
from actihealth.transform.sleep import DownSample85to30hz


def calc_counts(input_name, ouput_name, epoch_len=60):

    reading = RawNewcastlePSGReader().read(input_name)

    tmp_accel = reading.datasets["accelerometer"].data
    reading.datasets["accelerometer"].data = reading.datasets["accelerometer"].data[
        ["x", "y", "z"]
    ]

    down_samp = DownSample85to30hz()
    reading = down_samp.apply(reading)

    counts = CountsXYZ(
        input_dataset="accel_30hz",
        epoch=pd.Timedelta(epoch_len, unit="seconds"),
        keep_start_time=True,
    )
    reading = counts.apply(reading)

    reading.datasets[
        "accelerometer"
    ].data = tmp_accel  # get full accelerometer data back

    reading.to_h5(out_file=ouput_name)


if __name__ == "__main__":
    calc_counts(snakemake.input.files, " ".join(snakemake.output), int(snakemake.params.epoch_len))
