import diff_cont
import lambda_cont
import libs.agent_infra as ai
import os
import json
import datetime
import itertools
import constants as Cs
import concurrent.futures

enviroments = ["cartpole"] #, "lunarlander"]
seeds = [101, 102, 103]
TEST_EVAL_EPS = 15

#function ecapsulating one run of the algorithm
def task_job(env, alg, args, s):
    df, pop = Cs.ALG_MAPPING[alg].main(args=args)
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses

#generalised blueprint for exparimentations
def experiment_template(name, algorithms, grid_variables, grid_attr_f, filename_f):
    BASE = ["--out_path", "./Data/Experiments/"+str(name), "--experiment"]
    TEST_EVAL_EPS = 15
    for en, alg, *grid_v in itertools.product(enviroments, algorithms,*grid_variables):
        stat_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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

"""
Grid search for the most representative evo lenght and fitness trial episodes
"""
episodes = [1, 2, 3, 4, 5, 6, 7]
generations = [3, 7, 10, 15, 20, 30]
def ep_vs_gen_experiment():
    def argument_f(en, grid_v, seed ):
        eps, gens = grid_v
        return [
                "--enviroment", str(en),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(seed),         
        ]
    
    def filename_f(en, alg, grid_v):
        eps, gens = grid_v
        return "%s,%s,g%i,e%i.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        en,
                                        gens,
                                        eps,
                                    )

    experiment_template("ep_gen", ["diff", "lambda"], [episodes, generations], argument_f, filename_f)

"""
Grid Search over evolution relevant parameters to find most represenative ones
"""
        
mutation_rate = [0.1, 0.3, 0.5, 0.8,] #diff & lambda
cross_rate = [0.3, 0.5, 0.7, 0.9] #diff & lambda
mutation_sigma = [0.1, 0.5, 1, 2] #lambda
mutation_method = ["uniform", "mean"] #lambda

def fit_grid_search(eps, gens):

    def diff_argument_f(en, grid_v, seed ):
        mr, cr = grid_v
        return [
                "--enviroment", str(en),
                "--mutation_rate", str(mr),
                "--cross_rate", str(cr),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(seed),         
        ]
    
    def diff_filename_f(en, alg, grid_v):
        mr, cr = grid_v
        return "%s,%s,cr%.2f,mr%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()), 
                                        en, 
                                        cr,
                                        mr,
                                    )

    experiment_template("fit_grid", ["diff"], [mutation_rate, cross_rate], diff_argument_f, diff_filename_f)







if __name__ == "__main__":
    ep_vs_gen_experiment()
    fit_grid_search(3, 20)