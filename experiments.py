import diff_cont
import lambda_cont
import cartpole
import os
import json
import datetime
import itertools
#BASE = ["--out_path", "./Data/Experiments/", "--experiment"]

ALG_MAPPING = {"diff": diff_cont, "lambda":lambda_cont}

episodes = [1, 2, 3, 4, 5, 6, 7]
generations = [3, 7, 10, 15, 20, 30]

def ep_vs_gen_experiment():
    BASE = ["--out_path", "./Data/Experiments/ep_gen", "--experiment"]
    TEST_EVAL_EPS = 15
    for alg, eps, gens in itertools.product(["diff", "lambda"], episodes, generations):
        stats = {}
        for s in seeds:
            arguments = BASE + ["--episodes", str(eps), "--generations", str(gens)]
            args = ALG_MAPPING[alg].parser.parse_args(arguments)
            df, pop = ALG_MAPPING[alg].main(args=args)
            fitnesses = list(map(lambda x: cartpole.evalutation_b(x, 42, TEST_EVAL_EPS), pop))
            stats[s] = fitnesses
        dirpath = os.path.join(os.path.realpath(args.out_path), args.container,alg)
        filename = "%s,g%i,e%i.json" %  (
                                            str(datetime.datetime.utcnow()), 
                                            gens,
                                            eps,
                                        )
        json_path = os.path.join(dirpath, filename)
        with open(json_path, "w") as json_file:
            json.dump(stats,json_file)

def lambda_grid_search():
    pass  

def diff_grid_search():
    pass

if __name__ == "__main__":
    ep_vs_gen_experiment()