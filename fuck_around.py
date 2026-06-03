import itertools
import concurrent.futures
import constants as Cs
import os
import json
import datetime
import pickle
import optuna
from lunarlander import LunarLanderAgent

TEST_EVAL_EPS = 5
SEEDS = [101,102,103]
en = "lunarlander"
filename = "/home/schkliba/git/MastersThesis/Data/final/lunarlander/fit_archiving/lambda/2026-06-01_14-54-10_fit_archiving_cross_method: uniform|l: 70|m: 70|mr: 0.01|cr: 0.9000000000000001|mutation_sigma: 2.5|archiving_period: 4|archive_batch: 2|cross_uni: 0.8|ng: 15.plk"

with open(filename, "rb") as f:
    pops = pickle.load(f)

env = Cs.ENIVROMENTS[en](replay=True)

for p in pops:
    pop = pops[p]
    for i in pop:
        fit, beh = env.evalutation_b(i, p, TEST_EVAL_EPS)
        print(fit)
