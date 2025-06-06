# Conv-LSTM CDOM Prediction Model

This repository contains the Python code used to predict CDOM (Colored Dissolved Organic Matter) for the year 2024 using EOS-06 (OCM-3) satellite data. Because the raw NetCDF files are large and restricted to ISRO’s workspace, only the scripts are included here—you must place or download the NetCDF data yourself.

---

## 📂 Folder Structure
CONV-LSTM-for-CDOM-Modeling/
│
├── README.md
├── LICENSE # (MIT)
├── .gitignore
├── requirements.txt
│
├── data/
│ ├── raw/ ← Place your EOS-06 NetCDF files here (ignored by Git)
│ └── processed/ ← Intermediate .npy or monthly splits (ignored by Git)
│
├── src/ ← Python scripts 
│ ├── 1)cdom_derivation.py
│ ├── 2)npy_gen_0.01.py
│ ├── 3)range_processing.py
│ ├── 4)splits_0.01.py
│ ├── 5)conv_lstm_model_train.py
│ ├── 6)conv_lstm_model_test.py
│ └── 7)pred_to_netcdf.py
