import argparse
import glob
import os
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg_score-dir", type=str, default="/")
    args = parser.parse_args()

    models = [
        "mistral-7b-instruct-v0.2"
    ]
    
    avg_score = {}
    
    for model in models:
        avg_score[model] = {}
        dirpath = os.path.join(args.avg_score_dir, f"{model}/synthetic")
        for filename in glob.glob(f"{dirpath}/**/pred/summary.csv", recursive=True):
            seqlen = int(filename.split("/")[-3])
            with open(filename, "r") as f:
                df = pd.read_csv(f) 
                avg = df.iloc[1][1:].astype(float).mean()
                avg_score[model][seqlen] = avg
                print(filename)
                print(df.to_string(index=False, header=False))
    
    avg_score = {k: dict(sorted(v.items(), key=lambda x: x[0])) for k, v in avg_score.items()}
    print("average score", avg_score)
