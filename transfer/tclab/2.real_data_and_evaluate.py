import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
# import tclab


# print(tclab.find_arduinos())

# port1 = input("Enter port for Arduino 1: ")
# port2 = input("Enter port for Arduino 2: ")
port1 = "COM3"
port2 = "COM6"

rng = np.random.default_rng(42)

q1_time = 60
q2_time = 300
q1_setting = 0
q2_setting = 0

data_dir = Path("data")
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)

# ---

#with (
#    tclab.TCLab(port=port1) as lab1,
#    tclab.TCLab(port=port2) as lab2,
#):
# with tclab.TCLab(port=port1) as lab1:
#     for t in tclab.clock(period=48 * 3600, step=10, tol=0.1):
#         # lab1.Q1(100 * random.random())
#         # lab1.Q2(100 * random.random())
#         print(
#             f"t={t:5.1f}  T1={lab1.T1:5.1f}  T2={lab1.T2:5.1f}  Q1={q1_setting:3d}  Q2={q2_setting:3d}",#  Ta1={lab2.T1:5.1f}  Ta2={lab2.T2:5.1f}",
#             end="\r",
#         )

#         if t >= q1_time:
#             q1_time = t + rng.integers(6, 60) * 10  # 1-10 minutes
#             q1_setting = rng.integers(0, 101)  # 0-100
#             # 50% change of q1_setting being 0
#             if rng.random() < 0.5:
#                 q1_setting = 0
#             lab1.Q1(q1_setting)
#         if t >= q2_time:
#             q2_time = t + rng.integers(6, 60) * 10  # 1-10 minutes
#             q2_setting = rng.integers(0, 101)
#             # 50% change of q2_setting being 0
#             if rng.random() < 0.5:
#                 q2_setting = 0
#             lab1.Q2(q2_setting)
#         np.savez(
#             #str(data_dir.joinpath(f"{datetime.datetime.now().isoformat()}.npz")),
#             f"{datetime.datetime.now().isoformat()}.npz".replace(":",".."),
#             t=t,
#             T1=lab1.T1,
#             T2=lab1.T2,
#             Q1=q1_setting,
#             Q2=q2_setting,
#             # Ta1=lab2.T1,
#             # Ta2=lab2.T2,
#         )

# ---

data = []
import tqdm

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
