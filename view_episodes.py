from libs import constants as Cs
import pickle
import optuna

TEST_EVAL_EPS = 5
SEEDS = [101,102,103]
en = "cartpole"
filename = "/home/schkliba/git/MastersThesis/Data/final/cartpole/fitness/diff/server_try0/2026-06-23_05-08-04_fitness_l: 20|mr: 0.5|cr: 1.0|ng: 10.plk"

with open(filename, "rb") as f:
    pops = pickle.load(f)

env = Cs.ENIVROMENTS[en](replay=True)

for p in pops:
    pop = pops[p]
    for i in pop:
        fit, beh = env.evalutation_b(i, p, TEST_EVAL_EPS)
        print(fit)
