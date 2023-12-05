import numpy as np
from deap import tools

STAT_LIST = ["gen","min","avg","max","std"]

STAT_DEF = {
    "avg": np.mean,
    "min": np.min,
    "max": np.max,
    "std": np.std
}

def get_statistics(f):
    stat_obj = tools.Statistics(f)
    for stat in STAT_DEF:
        stat_obj.register(stat, STAT_DEF[stat])
    return stat_obj
