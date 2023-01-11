import os.path as op

from actihealth.reader.raw import RawNewcastlePSGReader
from actihealth.transform.sleep import AccelAngleZEpochMedian, VanHeesSIB
from utils import get_psg_downsampled


def sleep_classify_VH2015(
    file,
    output_fname,
    time_interval: int = 30,
    angle_thres: int = 5,
    time_thres_min: int = 5,
):

    accz = AccelAngleZEpochMedian(
        input_dataset="accelerometer", output="accel_epoch", resample_origin="start"
    )
    vh = VanHeesSIB(angle_thres=angle_thres, time_thres_min=time_thres_min)

    reading = RawNewcastlePSGReader().read(file)

    psg_results = get_psg_downsampled(reading, time_interval)

    # remove non xyz columns for VH
    reading.datasets["accelerometer"].data = reading.datasets["accelerometer"].data[
        ["x", "y", "z"]
    ]

    accz.apply(reading)
    vh.apply(reading)
    vh_Res = (
        reading.datasets["SIB_classification"]
        .data.resample(f"{time_interval}S", origin="start")
        .sum()
    )
    vh_Res = vh_Res.rename(columns={"is_sleep": "is_sleep_VH"})

    vh_Res_s = vh_Res.join(psg_results, how="right")
    assert vh_Res_s.shape[0] == psg_results.size

    vh_Res_s.to_csv(op.join(output_fname))


if __name__ == "__main__":
    sleep_classify_VH2015(
        snakemake.input.files,
        snakemake.output[0],
        int(snakemake.params.vh_time_interval),
        int(snakemake.params.vh_angle_thres),
        int(snakemake.params.vh_time_thres),
    )
