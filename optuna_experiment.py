
from optuna.samplers import TPESampler
from optuna.trial import Trial
from optuna.study import create_study
import argparse
from libs import constants as Cs
import numpy as np
from concurrent.futures import ProcessPoolExecutor

SEEDS = [101, 102, 103]
TEST_EVAL_EPS = 6

parser = argparse.ArgumentParser(
    description="Run final examination generation."
)

parser.add_argument(
    "--environment",
    "-e",
    type=str,
    required=True,
    help="Environment name"
)

parser.add_argument(
    "--algorithm",
    "-a",
    type=str,
    required=True,
    help="Algorithm name"
)

parser.add_argument(
    "--container",
    "-c",
    type=str,
    required=True,
    help="Container name"
)

parser.add_argument(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Output path for generated results"
)
parser.add_argument(
    "--seeds",
    "-s",
    type=int,
    default=5,
    help="Output path for generated results"
)

parser.add_argument(
    "--source",
    "-R",
    default="optuna",
    help="Source of choice"
)

args = parser.parse_args()
environment = args.environment

def lambda_objective(trial:Trial):
    env = Cs.ENIVROMENTS[environment]()
    cross_method = trial.suggest_categorical("crossmethod", ["uniform", "mean"])
    if environment == "lunarlander":
        l = trial.suggest_categorical("lambda",[ 40, 50, 60, 70])
        m = trial.suggest_categorical("mu",[40, 50, 60, 70])
        mr = trial.suggest_float("mutation_rate", 0, 0.1, step=0.01)
        cr = trial.suggest_float("cross_rate", 0.3, 1, step=0.1)
        sigma = trial.suggest_float("sigma", 0.5, 3, step=0.5)
        archiving = trial.suggest_int("archiving_period", 2, 5)
        archive_batch = trial.suggest_int("archive_batch", 1, 5)
        cross_uni = None
        if cross_method == "uniform":
            cross_uni = trial.suggest_float("cross", 0.4, 0.6, step=0.1)
    else:
        l = trial.suggest_categorical("lambda",[10,20,30])
        m = trial.suggest_categorical("mu",[10,20,30])
        mr = trial.suggest_float("mutation_rate", 0, 0.1, step=0.01)
        cr = trial.suggest_float("cross_rate", 0.3, 1, step=0.1)
        sigma = trial.suggest_float("sigma", 0.5, 2, step=0.5)
        archiving = trial.suggest_int("archiving_period", 2, 5)
        archive_batch = trial.suggest_int("archive_batch", 1, 5)

        cross_uni = None
        if cross_method == "uniform":
            cross_uni = trial.suggest_float("cross", 0.1, 0.9, step=0.1) 
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [
            executor.submit(Cs.lambda_cont.argumented_function, 
                env=environment,
                container=args.container,
                cross_method=cross_method,
                ng = 15,
                l = l,
                m = m,
                cr = cr,
                mr = mr,
                mutation_sigma = sigma,
                archiving_period=archiving,
                archive_batch=archive_batch,
                cross_uni = cross_uni if cross_uni is not None else 0.5,
                seed = seed) for seed in SEEDS
        ]

        pops = [f.result()[1] for f in futures]
        fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for pop in pops for p in pop ]


    fitnesses = list(map(lambda x:x[0], fitnesses))
    return np.max(fitnesses)


def diff_objective(trial:Trial):
    environment = args.environment
    env = Cs.ENIVROMENTS[environment]()
    if environment == "lunarlander": 
        l = trial.suggest_categorical("lambda",[40, 50, 60, 70])
        mr = trial.suggest_float("mutation_rate", 0, 1, step=0.1)
        cr = trial.suggest_float("cross_rate", 0.3, 1, step=0.1)
        archiving = trial.suggest_int("archiving_period", 2, 5)
        archive_batch = trial.suggest_int("archive_batch", 1, 5)
    else:
        l = trial.suggest_categorical("pop",[5,10,15,20])
        mr = trial.suggest_float("mutation_rate", 0, 1, step=0.1)
        cr = trial.suggest_float("cross_rate", 0.3, 1, step=0.1)
        archiving = trial.suggest_int("archiving_period", 2, 5)
        archive_batch = trial.suggest_int("archive_batch", 1, 5)   
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [
            executor.submit(Cs.ALG_MAPPING[args.algorithm].argumented_function, env=environment,
                container=args.container,
                ng = 15,
                l = l,
                cr = cr,
                mr = mr,
                archiving_period=archiving,
                archive_batch=archive_batch,
                seed = seed) for seed in SEEDS
        ]

        pops = [f.result()[1] for f in futures]
        fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for pop in pops for p in pop ]
    #trial.set_user_attr("scores", fitnesses)
    fitnesses = list(map(lambda x:x[0], fitnesses))
    return np.max(fitnesses)

objectives={
    "diff":diff_objective,
    "lambda":lambda_objective
}
sampler = TPESampler(n_startup_trials=20)
diff_fita_study = create_study(
    direction="maximize", 
    study_name=args.name, 
    sampler=sampler, 
    storage=f"sqlite:///Data/optuna/{args.environment}/{args.container}/{args.algorithm}.db", 
    load_if_exists=True
)


diff_fita_study.optimize(objectives[args.algorithm], n_trials=160, n_jobs=5)
print(diff_fita_study.best_trials)