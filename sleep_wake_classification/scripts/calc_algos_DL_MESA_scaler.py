"""Run deep learning algorithms using tensorflow. 
    if tensorflow not installed:    
    for GPU : conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install tensorflow 
"""
import numpy as np
import pandas as pd
import joblib
import os.path as op

from tensorflow.keras.models import load_model
from utils import get_data, output_name


def DL_MESA(input_name, out_files, fname_model_files, scaler_file):

    counts_ = pd.read_csv(input_name)
    scaler = joblib.load(scaler_file)

    df_meta = counts_[['timestamp','subjid','sleepstate']]

    counts = counts_.copy()
    counts = counts.rename(columns={'zcounts': 'activity', "sleepstate": 'psg'})
    counts['activity'] = scaler.transform(counts[['activity']])

    for fname_model_file in fname_model_files:
        fname_model = op.split(fname_model_file)[-1]
        print(fname_model)
        model = load_model(fname_model_file)
        
        x_test, y_test = get_data(counts, 100)
        assert((y_test == df_meta['sleepstate']).all())
        x_test = np.reshape(x_test, x_test.shape + (1,))
        
        try:
            predictions_prob = model.predict(x_test)
        except ValueError:
            x_test = x_test[:,:53,:]
            predictions_prob = model.predict(x_test)
        df_meta[f'is_sleep_{fname_model}'] = np.argmax(predictions_prob, 1)
    
    for k, gr in df_meta.groupby('subjid'):
        gr.drop('subjid', axis=1, inplace=True)
        gr.reset_index(drop=True, inplace=True)
        gr.to_csv(
            output_name(k, out_files), index=False
        )


if __name__ == '__main__':
    DL_MESA(input_name=snakemake.input.file_in,
            out_files=snakemake.output,
            fname_model_files=snakemake.input.fname_models,
            scaler_file=snakemake.input.scaler_file)

