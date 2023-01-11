# Sleep-Wake-Classification
Analysis code for sleep-wake algorithm comparison. Algorithms predict sleep-wake based on wrist accelerometer data.

Compare the following sleep-wake classification algorithms:
- Cole-Kripke
- Sadeh
- Oakley
- Sazonov
- van Hees
- Sundarajan (random forest)
- Palotti-LSTM
- Palotti-CNN

Snakemake workflow manager was used to run the analysis (https://snakemake.readthedocs.io/en/stable/)

Data analysis was performed on the Newcastle Polysomnography data set (https://zenodo.org/record/1160410#.Y77fZ-zML1w)

The deep learning models used can be found here https://github.com/qcri/sleep_awake_benchmark

