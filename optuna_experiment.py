import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
from optuna.study import create_study
import diff_cont
import lambda_cont
import libs.agent_infra as ai
import os
import json
import datetime
import itertools
import constants as Cs
import concurrent.futures
import numpy as np

seeds = [101, 102, 103]
TEST_EVAL_EPS = 6

#function ecapsulating one run of the algorithm
def task_job(env, alg, args, s):
    df, pop = Cs.ALG_MAPPING[alg].main(args=args)
    print("Testing " + str(s))
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses


def experiment_template(name, en, container,algorithms, grid_variables, grid_attr_f, filename_f):
    BASE = ["--out_path", "./Data/Experiments/"+str(name), "--experiment", "--container", str(container)]
    return_data = {}
    for alg, *grid_v in itertools.product( algorithms,*grid_variables):
        stat_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            print("Launching " + str(alg) + "on Enviroment " + str(en) + " with " + str(grid_v))
            for s in seeds:
                env = Cs.ENIVROMENTS[en]()
                arguments = BASE + grid_attr_f(en, grid_v, seed=s)
                args = Cs.ALG_MAPPING[alg].parser.parse_args(arguments)
                future = executor.submit(task_job, alg=alg, env=env, args=args, s=s)
                stat_futures[future] = s

        stats = {}
        for future in concurrent.futures.as_completed(stat_futures):
            s = stat_futures[future]
            stats[s] = future.result()
        dirpath = os.path.join(os.path.realpath(args.out_path), args.container,alg)
        filename = filename_f(en, alg, grid_v)
        json_path = os.path.join(dirpath, filename)
        with open(json_path, "w") as json_file:
            json.dump(stats,json_file)
            print("Finished "+ filename)

        return_data[(alg, *grid_v)] = stats
    return return_data

def objective(trial:Trial):
    env = Cs.ENIVROMENTS["cartpole"]()
    cross_method = trial.suggest_categorical("crossmethod", ["uniform", "mean"])
    l = trial.suggest_categorical("lambda",[10,20,30])
    m = trial.suggest_categorical("mu",[10,20,30])
    mr = trial.suggest_float("mutation_rate", 0, 0.1, step=0.1)
    cr = trial.suggest_float("cross_rate", 0.1, 1, step=0.1)

    df, pop = Cs.lambda_cont.argumented_function(
        enviroment="cartpole",
        container="fitness",
        cross_method=cross_method,
        ng=15,
        l = l,
        m=m,
        cr=cr,
        mr=mr,
        seed=20
    )
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    fitnesses = list(map(lambda x:x[0], fitnesses))
    return np.max(fitnesses)


sampler = TPESampler()
study = create_study(sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=5)
print(study.best_trials)