"""Run deep learning algorithms using tensorflow. 
    if tensorflow not installed:    
    for GPU : conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install tensorflow 
"""
import numpy as np
import pandas as pd
import os.path as op
import os
import glob
from tensorflow.keras.models import load_model

import tensorflow as tf
from train_DACNN import get_sequence_data
from sklearn.metrics import confusion_matrix

def calc_metrics_MESA(fname_out, model_files):

    data = get_sequence_data("past_25+25")
    dftest = pd.DataFrame(data['ytest'], columns=['sleepstate'])

    for model_file in model_files:
        fname_model = op.split(model_file)[-1]
        model = load_model(model_file)
        data = get_sequence_data("past_25+25")
        try:
            predictions_prob = model.predict(data['xtest'])
        except ValueError:
            data = get_sequence_data("past_25+1")
            predictions_prob = model.predict(data['xtest'])
        dftest[f'is_sleep_{fname_model}'] = np.argmax(predictions_prob, 1)
    
    
    cols = dftest.columns
    pred_cols = [c for c in cols if c.startswith('is_sleep_')]
    for pred_col in pred_cols:
        msk = ~dftest['sleepstate'].isna() & ~dftest[pred_col].isna()

        labs = dftest.loc[msk, 'sleepstate'].values.astype(int)

        preds = dftest.loc[msk, pred_col].values
        preds[preds < 1] = 0
        preds[preds >= 1] = 1

        conf_mat = confusion_matrix(labs, preds, labels=[0,1])
        tn, fn, tp, fp = conf_mat[0,0], conf_mat[1,0], conf_mat[1,1], conf_mat[0,1]
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        sensitivity = tp / (tp + fn)  # percentage of sleep labels correctly predicted
        specificity = tn / (tn + fp)  # percentage of wake labels correctly predicted
        if tp != 0:
            precision = tp / (tp + fp)   # percentage of sleep predictions that were correct
        else:
            precision = 0
        if precision != 0:
            f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
        else:
            f1 = 0

        # assume WASO is all the wake in the prediciton
        waso_sec = (len(preds) - preds.sum()) * 30
        waso_min = waso_sec / (60 * 363)
        waso_sec_psg = (len(labs) - labs.sum()) * 30
        waso_min_psg = waso_sec_psg / (60 * 363)
        waso_abs_diff = abs(waso_min - waso_min_psg)

        sleep_efficiency = (preds.sum() / len(preds)) * 100
        sleep_efficiency_psg = (labs.sum() / len(labs)) * 100
        sleep_efficiency_abs_diff = abs(sleep_efficiency - sleep_efficiency_psg)

        if not os.path.exists(fname_out):
            with open(fname_out, "w") as f:
                f.write("col_name,acc,sens,spec,prec,waso_min,waso_abs_diff,SE,SE_abs_diff,f1,TN,FN,TP,FP,waso_min_psg,sleep_efficiency_psg\n")
        with open(fname_out, "a+") as f:
            f.write(f"{pred_col},{accuracy},{sensitivity},{specificity},{precision},{waso_min},{waso_abs_diff},{sleep_efficiency},{sleep_efficiency_abs_diff},{f1},{tn},{fn},{tp},{fp},{waso_min_psg},{sleep_efficiency_psg}\n")


if __name__ == '__main__':
    calc_metrics_MESA(
        fname_out=snakemake.output[0],
        model_files=snakemake.input.fname_models,
    )