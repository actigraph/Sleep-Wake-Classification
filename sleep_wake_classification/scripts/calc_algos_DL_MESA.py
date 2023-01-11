"""Run deep learning algorithms using tensorflow. 
    if tensorflow not installed:    
    for GPU : conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install tensorflow 
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import get_data_old, output_name


def DL_MESA(
    input_name,
    out_files,
    NN_TYPES,
    SEQ_LENS,
    col=["xcounts", "ycounts", "zcounts"],
    root=6,
):

    counts_ = pd.read_csv(input_name)
    if len(col) > 1:
        counts_["VM"] = ((counts_[col] ** 2).sum(axis=1)) ** (1 / root)
    else:
        counts_["VM"] = (counts_[col]) ** (1 / root)
    df_meta = counts_[["timestamp", "subjid", "sleepstate"]]

    for NN_TYPE in NN_TYPES:
        for SEQ_LEN in SEQ_LENS:
            MODEL_OUTFILE = f"model_{NN_TYPE}_task1_raw_seq{SEQ_LEN}.pkl"
            model = load_model("models_palotti/" + MODEL_OUTFILE)

            counts = counts_.copy()
            counts = counts.rename(columns={"VM": "activity", "sleepstate": "psg"})
            counts["activity"] = (
                counts["activity"] - counts["activity"].mean()
            ) / counts["activity"].std()

            x_test, y_test = get_data_old(counts, SEQ_LEN)
            assert (y_test == df_meta["sleepstate"]).all()
            x_test = np.reshape(x_test, x_test.shape + (1,))

            predictions_prob = model.predict(x_test)
            df_meta[f"is_sleep_{NN_TYPE}_{SEQ_LEN}"] = np.round(
                predictions_prob
            ).astype(int)

    for k, gr in df_meta.groupby("subjid"):
        gr.drop("subjid", axis=1, inplace=True)
        gr.reset_index(drop=True, inplace=True)
        gr.to_csv(
            output_name(k, out_files), index=False
        )


if __name__ == "__main__":
    input_name = snakemake.input.file_in
    out_files = snakemake.output
    NN_TYPES = snakemake.params.NN_TYPE
    SEQ_LENS = snakemake.params.SEQ_LEN
    cols = snakemake.params.DL_cols
    root = snakemake.params.DL_root

    NN_TYPES = NN_TYPES.split(",")
    SEQ_LENS = [int(n) for n in SEQ_LENS.split(",")]
    col = cols.split(",")
    DL_MESA(input_name, out_files, NN_TYPES, SEQ_LENS, col=col, root=int(root))
