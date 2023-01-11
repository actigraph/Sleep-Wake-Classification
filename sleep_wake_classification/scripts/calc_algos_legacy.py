import numpy as np
import pandas as pd
from actihealth.data.core import Reading
from actihealth.transform.sleep import BioPsyKitSleep
from utils import get_psg_downsampled


def run_legacy_algos(input_name, output_name):

    reading = Reading.from_h5(input_name)

    psg_results = get_psg_downsampled(reading, 30)

    algo_list = ["sadeh", "cole_kripke", "sazonov", "scripps_clinic", "webster"]

    results_list = []
    for algo_name in algo_list:

        bpk_sleep = BioPsyKitSleep(
            input_dataset="countsxyz",
            output=algo_name,
            z_axis_only=True,
            classification_algo=algo_name,
        )
        reading = bpk_sleep.apply(reading)

        algo_col_name = f"is_sleep_{algo_name}"

        df = reading.datasets[algo_name].data.copy()
        df = df.rename(columns={"sleep_wake": algo_col_name})

        # if epoch is 1-min, then convert to 30sec and repeat every value once
        df_repeat = np.repeat(df[algo_col_name], 2)
        ts_30 = pd.date_range(start=df.index[0], end=df.index[-1], freq=f"30s")
        df_30 = pd.DataFrame(
            zip(df_repeat, ts_30), columns=[algo_col_name, "timestamp"]
        )
        df_30.set_index("timestamp", inplace=True, drop=True)

        results_list.append(df_30)

    df_final = pd.concat(results_list, axis=1)
    df_final["is_sleep_all_sleep"] = 1
    df_final["is_sleep_all_wake"] = 0

    df_final = df_final.join(psg_results, how="right")
    df_final.to_csv(output_name)


if __name__ == "__main__":
    run_legacy_algos(snakemake.input[0], snakemake.output[0])
