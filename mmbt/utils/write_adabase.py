import pandas as pd
import h5py
import math
import pyarrow as pa
import os
import numpy as np
from tqdm import tqdm
from mmbt.utils.get_filenames import GetSubFiles


print(os.getcwd())
split = 'val'
load_path = r'F:\09_proj_multi_modality\multi-modality-biosignal\datasets\ADABase\ADABase_' + split + '.h5'

save_path = os.path.dirname(load_path)
mat_data = h5py.File(load_path, 'r+')
subset = mat_data['data']

ecgs = subset['ecg'].squeeze(0).transpose()
ppgs = subset['ppg'].squeeze(0).transpose()
labels = subset['label'].squeeze(0)

data_list = []
for i in tqdm(range(ecgs.shape[0])):
    ecg = ecgs[i]
    has_nan = np.isnan(ecg).any()
    if has_nan:
        continue
    ppg = ppgs[i]
    has_nan = np.isnan(ppg).any()
    if has_nan:
        continue
    label = labels[i]
    data = (ecg, ppg, label)
    data_list.append(data)

dataframe = pd.DataFrame(
    data_list,
    columns=[
        "ECG",
        "PPG",
        "CL"
    ]
)


table = pa.Table.from_pandas(dataframe)

os.makedirs(save_path, exist_ok=True)
with pa.OSFile(f"{save_path}/ADABase_{split}.arrow", "wb") as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)


