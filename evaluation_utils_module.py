import numpy as np
from constants import EXAMPLE_MAPPING
import constants as Cs
import optuna
import json
from pathlib import Path
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

def select_minimal_examples(examples):
    rates = -np.inf
    pop = np.inf
    selected_trials = []
    selected_trials_index = []
    for i, t in enumerate(examples):
        k = t["cross_rate"] + t["mutation_rate"]
        if rates == k:
            selected_trials.append(t)
        elif rates < k:
            rates = k
            selected_trials = [t]
            selected_trials_index.append(i)
    refined_trials = []
    refined_trials_index = []       
    for i, t in zip(selected_trials_index, selected_trials):
        if "lambda" in t:
            k = t["lambda"]
            if "mu" in t:
                k+=t["mu"]
        elif "pop" in t:
            k=t["pop"]
        else: raise NameError(f"wtf")
        if pop == k:
            refined_trials.append(t)
        elif pop > k:
            pop = k
            refined_trials = [t]
            refined_trials_index.append(i)
    return refined_trials, refined_trials_index


def select_minimal_examples_old(examples):
    pop = np.inf
    refined_trials = []
    refined_trials_index = []       
    for i, t in enumerate(examples):
        if "lambda" in t:
            k = t["lambda"]
            if "mu" in t:
                k+=t["mu"]
        elif "pop" in t:
            k=t["pop"]
        else: raise NameError(f"wtf")
        if pop == k:
            refined_trials.append(t)
        elif pop > k:
            pop = k
            refined_trials = [t]
            refined_trials_index.append(i)
    return refined_trials, refined_trials_index

def load_from_grid_search(en, container, alg):
    with open("relevant_grid_search.json", "r") as f:
        grid_doc = json.load(f)
    latest = grid_doc[en]
    storage = f"Data/grid_search/{latest}/{en}/{container}/{alg}"
    #study_name = grid_doc[container][alg][en]

    directory = Path(storage)
    print(directory)
    for json_file in directory.glob("*.json"):
        with open(json_file, "r") as f:
            protocol = json.load(f)
    most_promising  = protocol["final"]
    return most_promising

def load_from_optuna(en, container, alg):
    with open("relevant_studies.json", "r") as f:
        relevant_study_names = json.load(f)
    storage = f"sqlite:///Data/optuna/{en}/{container}/{alg}.db"
    study_name = relevant_study_names[container][alg][en]

    new_study = optuna.load_study(study_name=study_name,storage=storage)
    most_promising, indexes = select_minimal_examples([t.params for t in new_study.best_trials])

    return most_promising, indexes

def load_from_relevant_file(en, container, alg, filename):
    with open(filename, "r") as f:
        relevant_json_file = json.load(f)
        most_promising = relevant_json_file[container][alg][en]
    return most_promising

def load_from_relevant_file2(en, container, alg, filename):
    with open(filename, "r") as f:
        relevant_json_file = json.load(f)
        most_promising = relevant_json_file[container][en][alg]
    return most_promising
