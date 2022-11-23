import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv("SG2022_C39.d02", delimiter='\t')
sma_window = 200
HIT1_START = 191.0
HIT1_END = 191.35
HIT2_START = 504.5
HIT2_END = 504.75
# vals_of_interest_columns = ["Time (sec)", "Force (kN)", "Jaw (mm)"]
df = pd.read_csv("SG2022_C44.d01", delimiter='\t')
units = df.iloc[0]
new_columns = [col + " " + u if type(u) == str else col for col, u in zip(df.columns, units)]
df.columns = new_columns
df.drop(0, inplace=True)
df = df.astype(float)
# hit_1 = df[(HIT1_START <= df["Time (sec)"]) & (df["Time (sec)"] <= HIT1_END)]
hit_1 = df[(HIT2_START <= df["Time (sec)"]) & (df["Time (sec)"] <= HIT2_END)]
# hit_1 = hit_1[vals_of_interest_columns]
# hit_1 = hit_1.rolling(window=sma_window).mean()
hit_1_sma = hit_1.rolling(window=sma_window).mean()
plot_columns = ["Force (kN)", "Jaw (mm)", "TC1 (C)", "TC2 (C)"]
# plot_columns = ["Force (kN)", "Jaw (mm)"]
hit_1.plot("Time (sec)", plot_columns)
plt.grid()
hit_1_sma.plot("Time (sec)", plot_columns)
plt.grid()
hit_1_sma["Jaw (mm)"] = -hit_1_sma["Jaw (mm)"]
hit_1_sma["Force (kN)"] = -hit_1_sma["Force (kN)"]
hit_1_sma.plot("Jaw (mm)", "Force (kN)")
plt.grid()
plt.show()
pass