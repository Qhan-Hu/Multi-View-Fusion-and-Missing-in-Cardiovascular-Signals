import pandas as pd
import h5py
import math
import pyarrow as pa
import os
from tqdm import tqdm
from mmbt.utils.get_filenames import GetSubFiles


print(os.getcwd())
split = 'val'  # 'train' or 'val' or 'test'
load_path = './datasets/PulseDB/AAMI_Test_Subset.mat'
save_path = os.path.dirname(load_path)
concatenated_file_path = os.path.join(save_path,'PulseDB_val.arrow')
mat_data = h5py.File(load_path, 'r+')
subset = mat_data['Subset']
# print(subset.keys())

block_size = 10000
total_size = subset['Signals'].shape[2]
block_num = math.ceil(total_size/block_size)
offset = 0

for j in tqdm(range(block_num), desc='Outer loop'):
    data_list = []
    end = min(offset + block_size, total_size)
    size = end - offset

    for i in tqdm(range(offset, end), desc='Inner loop: block_'+str(j), mininterval=10, leave=False):
        ecg = subset['Signals'][:, 0, i]
        ppg = subset['Signals'][:, 1, i]
        abp = subset['Signals'][:, 2, i]
        age = subset['Age'][0][i]
        sbp = subset['SBP'][0][i]
        dbp = subset['DBP'][0][i]
        weight = subset['Weight'][0][i]
        height = subset['Height'][0][i]
        subject = ''.join(chr(c) for c in mat_data[mat_data['Subset']['Subject'][0][i]])
        gender = ''.join(chr(c) for c in mat_data[mat_data['Subset']['Gender'][0][i]])
        data = (subject, ecg, ppg, abp, sbp, dbp, age, gender, weight, height, split)
        data_list.append(data)

    dataframe = pd.DataFrame(
        data_list,
        columns=[
            "Subject",
            "ECG",
            "PPG",
            "ABP",
            "SBP",
            "DBP",
            "Age",
            "Gender",
            "Weight",
            "Height",
            "split"
        ]
    )
    table = pa.Table.from_pandas(dataframe)

    os.makedirs(save_path, exist_ok=True)
    with pa.OSFile(f"{save_path}/PulseDB_{split}_{j}.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    offset = end


arrow_subsets = GetSubFiles(save_path, 'arrow')

table_list = []
for i in tqdm(range(len(arrow_subsets)), desc='Concatenation Progress:'):
    arrow_subset_path = os.path.join(save_path, arrow_subsets[i])
    with pa.ipc.open_file(arrow_subset_path) as f:
        table = f.read_all()
        table_list.append(table)

concatenated_table = pa.concat_tables(table_list)
with pa.ipc.new_file(concatenated_file_path, concatenated_table.schema) as f:
    f.write_table(concatenated_table)
