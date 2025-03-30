import numpy as np

def hankelwts():
    with open(".\hankelwts_wt0.txt", "r") as file:
        wt0 =[float(line.strip()) for line in file]
    with open(".\hankelwts_wt1.txt", "r") as file:
        wt1 =[float(line.strip()) for line in file]
    return np.array(wt0), np.array(wt1)