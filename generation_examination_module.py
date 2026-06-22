import itertools
import concurrent.futures
import constants as Cs
import os
import json
import datetime
import pickle
from pathlib import Path
from evaluation_utils_module import  rename, task_job, load_from_grid_search, load_from_optuna

# lambda fit archving [FrozenTrial(number=80, state=<TrialState.COMPLETE: 1>, values=[118.11550385203829], datetime_start=datetime.datetime(2026, 5, 22, 5, 4, 53, 822616), datetime_complete=datetime.datetime(2026, 5, 22, 5, 36, 8, 359073), params={'crossmethod': 'uniform', 'lambda': 70, 'mu': 70, 'mutation_rate': 0.01, 'cross_rate': 0.9000000000000001, 'sigma': 0.5, 'archiving_period': 4, 'archive_batch': 5, 'cross': 0.6}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.1, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5), 'archiving_period': IntDistribution(high=5, log=False, low=2, step=1), 'archive_batch': IntDistribution(high=5, log=False, low=1, step=1), 'cross': FloatDistribution(high=0.9, log=False, low=0.1, step=0.1)}, trial_id=81, value=None), FrozenTrial(number=86, state=<TrialState.COMPLETE: 1>, values=[118.11550385203829], datetime_start=datetime.datetime(2026, 5, 22, 5, 36, 31, 908497), datetime_complete=datetime.datetime(2026, 5, 22, 6, 11, 5, 128793), params={'crossmethod': 'uniform', 'lambda': 70, 'mu': 70, 'mutation_rate': 0.01, 'cross_rate': 0.9000000000000001, 'sigma': 0.5, 'archiving_period': 4, 'archive_batch': 5, 'cross': 0.6}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.1, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5), 'archiving_period': IntDistribution(high=5, log=False, low=2, step=1), 'archive_batch': IntDistribution(high=5, log=False, low=1, step=1), 'cross': FloatDistribution(high=0.9, log=False, low=0.1, step=0.1)}, trial_id=87, value=None)]
# lambda - this one is bad actually novelty [FrozenTrial(number=4, state=<TrialState.COMPLETE: 1>, values=[-9.291787437438707], datetime_start=datetime.datetime(2026, 5, 19, 22, 25, 13, 886515), datetime_complete=datetime.datetime(2026, 5, 19, 23, 0, 16, 677521), params={'crossmethod': 'mean', 'lambda': 60, 'mu': 60, 'mutation_rate': 0.18, 'cross_rate': 1.0, 'sigma': 2.5}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.5, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5)}, trial_id=208, value=None)]
GENS = {
    "cartpole": [5, 10, 15, 20, 25],
    "lunarlander":[10, 15, 25, 35, 50]
}

def evaluation_of_setup(en, alg, container, generations, experiment_name, out_path, seeds, type_out, **kwargs):
    #we do not deviate the sigma
    # first we evaluate the current setup
    print(generations)
    stat_futures = {}
    gens_future={}
    args = rename(kwargs)
    dirpath = os.path.join(os.path.realpath(out_path), en,container,alg, experiment_name)

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        print("Launching " + alg + "on Enviroment " + str(en))
        for ng, s in itertools.product(generations, seeds):
            ags = args.copy()
            ags["ng"] = ng
            future = executor.submit(task_job, container=container, alg=alg, en=en, args=ags, s=s, out_path=dirpath + "/stat")
            stat_futures[future] = s
            gens_future[future] = ng

    stats = {}
    for future in concurrent.futures.as_completed(stat_futures):
        s = stat_futures[future]
        ng = gens_future[future]
        result = future.result()
        if "fitness" not in stats:
            stats["fitness"] = {}
        if str(ng) not in stats["fitness"]:
            stats["fitness"][str(ng)] = {}
        stats["fitness"][str(ng)][s] = result[0]
    
    if type_out == "base":
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}_{container}_"+ "|".join(f"{k}: {v}" for k, v in args.items())
        stats["environment"] = en
        stats["algorithm"] = alg
        stats["container"] = container
        stats["arguments"] = args
        json_path = os.path.join(dirpath, filename + ".json")
        pickle_path = os.path.join(dirpath, filename + ".plk")

        with open(json_path, "w") as json_file:
            json.dump(stats,json_file)
            print("Finished "+ filename)
        # with open(pickle_path, "wb") as f:
            # pickle.dump(pops, f)
        
    else:
        path = f"./Data/generation_ex/{en}/{container}/{alg}/{experiment_name}"
        json_files = list(Path(path).glob("*.json"))
        print(f"Found {json_files[0]}")
        with open(json_files[0], "r+") as f:
            data = json.load(f)
            print([i for i  in data["fitness"]])
            data["fitness"] = {**data["fitness"],  **stats["fitness"]}
            print([i for i  in data["fitness"]])
            f.seek(0)       
            json.dump(data, f, indent=4)
            f.truncate()
        print(f"Done {json_files[0]}")
    return stats

def make_gen_examination(en, alg, container ,name, seeds, source="optuna", type_out="base", generations = []):
    if source == "optuna":
        most_promising, _ = load_from_optuna(en=en, alg=alg, container=container)
        most_promising = most_promising[0]
    elif source == "grid_search":
        most_promising = load_from_grid_search(en=en, alg=alg, container=container)

    
    stats = evaluation_of_setup(
        en=en, 
        alg=alg, 
        container=container,
        generations = GENS[en] if type_out == "base" else generations,
        experiment_name=name,
        type_out=type_out,
        seeds=seeds,
        out_path=f"./Data/generation_ex",
        **most_promising
    )
    return stats


