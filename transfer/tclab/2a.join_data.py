
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm

data = []

data_csv = Path(r"E:\TCLab\data\data.csv")
pd_data = pd.read_csv(data_csv)
for index, row in pd_data.iterrows():
    data.append(row.to_dict())
data_dir = Path(r"E:\TCLab")
for file in tqdm.tqdm(data_dir.rglob("*.npz")):
    d = dict(np.load(file))
    d["dt"] = file.stem
    data.append(d)
df = pd.DataFrame(data)
df.to_csv(data_dir.joinpath("data.csv"), index=False)