import os
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--i", required=True, type=str)
args = parser.parse_args()

assert os.path.isfile(args.i)

df = []
for line in open(args.i).readlines():
    if line[0].isdigit() and "\t" in line:
        data = line.strip().replace(" ","").split("\t")
        df.append(data)

df = pd.DataFrame(df, dtype=float)
print(df.tail(5))
print(df.sort_values(2, ascending=False).head(5))
print(df.sort_values(6, ascending=False).head(5))
