import os

import pandas as pd
from sklearn.metrics import confusion_matrix


def calc_metrics(
    fname_in,
    fname_out,
    lab_col="sleepstate",
    pred_col_prefx="is_sleep",
    ncount_2_sleep=1,
):

    preds_df = pd.read_csv(fname_in)

    cols = preds_df.columns
    pred_cols = [c for c in cols if c.startswith(pred_col_prefx)]
    for pred_col in pred_cols:
        msk = ~preds_df[lab_col].isna() & ~preds_df[pred_col].isna()

        labs = preds_df.loc[msk, lab_col].values.astype(int)

        preds = preds_df.loc[msk, pred_col].values
        preds[preds < ncount_2_sleep] = 0
        preds[preds >= ncount_2_sleep] = 1

        conf_mat = confusion_matrix(labs, preds, labels=[0, 1])
        tn, fn, tp, fp = conf_mat[0, 0], conf_mat[1, 0], conf_mat[1, 1], conf_mat[0, 1]
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        sensitivity = tp / (tp + fn)  # percentage of sleep labels correctly predicted
        specificity = tn / (tn + fp)  # percentage of wake labels correctly predicted
        if tp != 0:
            precision = tp / (
                tp + fp
            )  # percentage of sleep predictions that were correct
        else:
            precision = 0
        if precision != 0:
            f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
        else:
            f1 = 0

        # assume WASO is all the wake in the prediciton
        waso_sec = (len(preds) - preds.sum()) * 30
        waso_min = waso_sec / 60
        waso_sec_psg = (len(labs) - labs.sum()) * 30
        waso_min_psg = waso_sec_psg / 60
        waso_abs_diff = abs(waso_min - waso_min_psg)

        #
        tst_sec = (len(preds) * 30) - waso_sec
        tst_min = tst_sec / 60
        tst_sec_psg = (len(labs) * 30) - waso_sec_psg
        tst_min_psg = tst_sec_psg / 60
        tst_abs_diff = abs(tst_min - tst_min_psg)

        sleep_efficiency = (preds.sum() / len(preds)) * 100
        sleep_efficiency_psg = (labs.sum() / len(labs)) * 100
        sleep_efficiency_abs_diff = abs(sleep_efficiency - sleep_efficiency_psg)

        if ncount_2_sleep > 1:
            pred_col += f"_n2sleep_{ncount_2_sleep}"
        _, fname = os.path.split(fname_in)
        if not os.path.exists(fname_out):
            with open(fname_out, "w") as f:
                f.write(
                    "file,col_name,acc,sens,spec,prec,waso_min,waso_abs_diff,SE,SE_abs_diff,f1,TN,FN,TP,FP,"
                    "waso_min_psg,sleep_efficiency_psg,tst_min,tst_min_psg,tst_abs_diff\n"
                )
        with open(fname_out, "a+") as f:
            f.write(
                f"{fname},{pred_col},{accuracy},{sensitivity},{specificity},{precision},{waso_min},{waso_abs_diff},"
                f"{sleep_efficiency},{sleep_efficiency_abs_diff},{f1},{tn},{fn},{tp},{fp},{waso_min_psg},"
                f"{sleep_efficiency_psg},{tst_min},{tst_min_psg},{tst_abs_diff}\n"
            )


if __name__ == "__main__":
    for ith_ncount_2_sleep in str(snakemake.params.ncount_2_sleep).split(","):
        calc_metrics(
            snakemake.input[0], snakemake.output[0],
            snakemake.params.lab_col, snakemake.params.pred_col_prefx, int(ith_ncount_2_sleep)
        )
