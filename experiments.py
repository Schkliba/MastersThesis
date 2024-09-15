import diff_cont
import lambda_cont
import libs.agent_infra as ai
import os
import json
import datetime
import itertools
import constants as Cs
import concurrent.futures
#BASE = ["--out_path", "./Data/Experiments/", "--experiment"]

enviroments = ["cartpole"] #, "lunarlander"]
episodes = [1, 2, 3, 4, 5, 6, 7]
generations = [3, 7, 10, 15, 20, 30]
seeds = [101, 102, 103]

def ep_vs_gen_experiment():
    BASE = ["--out_path", "./Data/Experiments/ep_gen", "--experiment"]
    TEST_EVAL_EPS = 15
    for en, alg, eps, gens in itertools.product(enviroments, ["diff", "lambda"], episodes, generations):
        stats = {}
        env = Cs.ENIVROMENTS[en]()
        for s in seeds:
            arguments = BASE + [
                "--enviroment", str(en),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(s),
            ]
            args = Cs.ALG_MAPPING[alg].parser.parse_args(arguments)
            df, pop = Cs.ALG_MAPPING[alg].main(args=args)

            fitnesses = list(map(lambda x: env.evalutation_b(x, 42, TEST_EVAL_EPS), pop))
            stats[s] = fitnesses
            print("Finished seed: " + str(s))
        dirpath = os.path.join(os.path.realpath(args.out_path), args.container,alg)
        filename = "%s,g%i,e%i.json" %  (
                                            alg + "_" + en + str(datetime.datetime.utcnow()), 
                                            gens,
                                            eps,
                                        )
        json_path = os.path.join(dirpath, filename)
        with open(json_path, "w") as json_file:
            json.dump(stats,json_file)
        print("Finished "+ filename)



def task_job(env, alg, arguments, s):
    args = Cs.ALG_MAPPING[alg].parser.parse_args(arguments)
    df, pop = Cs.ALG_MAPPING[alg].main(args=args)
    fitnesses = list(map(lambda x: env.evalutation_b(x, 42, TEST_EVAL_EPS), pop))
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses


        
mutation_rate = [0.1, 0.3, 0.5, 0.8,]
cross_rate = [0.3, 0.5, 0.7, 0.9]
def diff_grid_search():

    BASE = ["--out_path", "./Data/Experiments/diff_grid", "--experiment"]
    TEST_EVAL_EPS = 15
    alg = "diff"
    for en, eps, gens in itertools.product(enviroments, episodes, generations):
        stats = {}
        env = Cs.ENIVROMENTS[en]()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            arguments = BASE + [
                "--enviroment", str(en),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(s),
            ]
            for s in seeds:
                stats[s] = executor.submit(task_job, alg=alg, env=env, arguments=arguments, s=s)
                #args = Cs.ALG_MAPPING[alg].parser.parse_args(arguments)
                #df, pop = Cs.ALG_MAPPING[alg].main(args=args)

            #   fitnesses = list(map(lambda x: env.evalutation_b(x, 42, TEST_EVAL_EPS), pop))

        dirpath = os.path.join(os.path.realpath(args.out_path), args.container,alg)
        filename = "%s,g%i,e%i.json" %  (
                                            alg + "_" + en + str(datetime.datetime.utcnow()), 
                                            gens,
                                            eps,
                                        )
        json_path = os.path.join(dirpath, filename)
        with open(json_path, "w") as json_file:
            json.dump(stats,json_file)
        print("Finished "+ filename)



def lambda_grid_search():
    pass  



if __name__ == "__main__":
    ep_vs_gen_experiment()