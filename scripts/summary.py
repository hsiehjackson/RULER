import glob
import pandas as pd

models = [
    "mistral-7b-instruct-v0.2"
]

result = {}

for model in models:
    result[model] = {}
    dirpath = f"/mnt/workdisk/ying/RULER/results/{model}/synthetic"
    for filename in glob.glob(f"{dirpath}/**/pred/summary.csv", recursive=True):
        seqlen = filename.split("/")[-3]
        with open(filename, "r") as f:
            df = pd.read_csv(f) 
            avg = df.iloc[1][1:].astype(float).mean()
            result[model][seqlen] = avg

print(result)

