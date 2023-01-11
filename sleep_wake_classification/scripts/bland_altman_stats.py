import pandas as pd
from utils import BlandAltman


def bland_altman_stats(fname_in, fname_out_waso, fname_out_se, fname_out_tst):
    df_full = pd.read_csv(fname_in)

    algo_list = df_full.col_name.unique()

    for idx, i_algo in enumerate(algo_list):
        df_set = df_full[df_full.col_name == i_algo]
        ba_se = BlandAltman(df_set.sleep_efficiency_psg, df_set.SE)
        se_stats = ba_se.return_stats()
        new_df_se = pd.DataFrame(se_stats, index=[i_algo])

        ba_waso = BlandAltman(df_set.waso_min_psg, df_set.waso_min)
        waso_stats = ba_waso.return_stats()
        new_df_waso = pd.DataFrame(waso_stats, index=[i_algo])

        print(df_set.columns)
        ba_tst = BlandAltman(df_set.tst_min_psg, df_set.tst_min)
        tst_stats = ba_tst.return_stats()
        new_df_tst = pd.DataFrame(tst_stats, index=[i_algo])

        if idx == 0:
            df_out_se = new_df_se
            df_out_waso = new_df_waso
            df_out_tst = new_df_tst
        else:
            df_out_se = pd.concat([df_out_se, new_df_se])
            df_out_waso = pd.concat([df_out_waso, new_df_waso])
            df_out_tst = pd.concat([df_out_tst, new_df_tst])

    # calculate CI width
    df_out_se["CI_width"] = df_out_se["CI_95%+"] - df_out_se["CI_95%-"]
    df_out_waso["CI_width"] = df_out_waso["CI_95%+"] - df_out_waso["CI_95%-"]
    df_out_tst["CI_width"] = df_out_tst["CI_95%+"] - df_out_tst["CI_95%-"]

    df_out_se.to_csv(fname_out_se)
    df_out_waso.to_csv(fname_out_waso)
    df_out_tst.to_csv(fname_out_tst)


if __name__ == "__main__":
    bland_altman_stats(snakemake.input.fname, snakemake.output.output_waso,
                       snakemake.output.output_se, snakemake.output.output_tst)
