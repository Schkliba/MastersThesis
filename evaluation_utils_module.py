import numpy as np
from constants import EXAMPLE_MAPPING
import constants as Cs

TEST_EVAL_EPS = 6

def rename(dict):
    return {EXAMPLE_MAPPING.get(k, k): v for k, v in dict.items()}

def task_job(en, alg, container, args, s, out_path):
    env = Cs.ENIVROMENTS[en]()
    df, pop = Cs.ALG_MAPPING[alg].argumented_function(env=en, container=container, seed=s, out_path=out_path, **args)
    print("Testing " + str(s))
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses, pop

def select_minimal_exaples(examples):
    pop = np.inf
    selected_trials = []
    selected_trials_index = []
    for i, t in enumerate(examples):
        if "lambda" in t:
            k = t["lambda"]
            if "mu" in t:
                k+=t["mu"]
        elif "pop" in t:
            k=t["pop"]
        else: raise NameError(f"wtf")
        if pop == k:
            selected_trials.append(t)
        elif pop > k:
            pop = k
            selected_trials = [t]
            selected_trials_index.append(i)
    return selected_trials, selected_trials_index