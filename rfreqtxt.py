import pandas as pd
import sys

def remove_channel(package):
    if package[0] == "#":
        return package
    else:
        second_equal = len(package) - package[::-1].find("=")
        return package[:second_equal-1]

def main():
    path = sys.argv[1]

    try:
        df = pd.read_table(path, names=["#package"], encoding="UTF-16")
    except:
        df = pd.read_table(path, names=["#package"], encoding="UTF-8")
    df.iloc[:, 0] = df.iloc[:,0].map(remove_channel)
    df.to_csv(path, index=None, columns=None)

if __name__ == "__main__":
    main()