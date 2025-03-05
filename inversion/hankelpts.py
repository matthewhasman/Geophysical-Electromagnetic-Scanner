import numpy as np

def hankelpts():
    with open(".\hankelpts_values.txt", "r") as file:
        vals = [float(line.strip()) for line in file]
    return np.array(vals)