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
TEST_EVAL_EPS = 6

#function ecapsulating one run of the algorithm
def task_job(env, alg, args, s):
    df, pop = Cs.ALG_MAPPING[alg].main(args=args)
    print("Testing " + str(s))
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses

#generalised blueprint for exparimentations
def experiment_template(name, en, container,algorithms, grid_variables, grid_attr_f, filename_f):
    BASE = ["--out_path", "./Data/Experiments/"+str(name), "--experiment", "--container", str(container)]
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
        
mutation_rate = {
    "cartpole": [0.1, 0.3, 0.5, 0.8,]
} #diff & lambda
cross_rate = {
    "cartpole": [0.3, 0.5, 0.7, 0.9]
} #diff & lambda
mutation_method = ["uniform", "mean"] #lambda

    
def lambda_mutation_search(eps, gens, en):
    
    def lambda_argument_f(en, grid_v, seed):
        mr, cr, method, sigma = grid_v
        return [
                "--enviroment", str(en),
                "--mutation_rate", str(mr),
                "--mutation_sigma", str(sigma),
                "--cross_method", str(method),
                "--cross_rate", str(cr),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        mr, cr, method, sigma = grid_v

        return "%s,%s,%s,cr%.2f,mr%.2f,ms%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        method, 
                                        en, 
                                        cr,
                                        mr,
                                        sigma
                                    )

    experiment_template(
        "fit_grid",
        en,
        "fitness", 
        ["lambda"], 
        [mutation_rate[en], cross_rate[en], mutation_method], 
        lambda_argument_f, 
        lambda_filename_f
    )
    
    experiment_template(
        "nov_grid",
        en,
        "novelty", 
        ["lambda"], 
        [mutation_rate[en], cross_rate[en], mutation_method], 
        diff_argument_f, 
        diff_filename_f
    )

def diff_mutation_search(eps, gens, en):

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
    experiment_template(
        "fit_grid",
        en,
        "fitness", 
        ["diff"], 
        [mutation_rate[en], cross_rate[en]], 
        diff_argument_f, 
        diff_filename_f
    )

    experiment_template(
        "nov_grid",
        en,
        "novelty", 
        ["diff"], 
        [mutation_rate[en], cross_rate[en]], 
        diff_argument_f, 
        diff_filename_f
    )

lambdas_ = {
    "cartpole": [5, 10, 20, 30, 40]
}

mus = {
    "cartpole": [5, 10, 20, 30, 40]
}

def lambda_mu_mut_search(en, eps, gen, lam):
    
    def lambda_argument_f(en, grid_v, seed):
        mu, mr = grid_v
        return [
                "--enviroment", str(en),
                #"--cross_method", str(method),
                "--mutation_rate", str(mr),
                "--cross_rate", str(0),
                "--mu", str(mu),
                "--lambdan", str(lam),
                "--episodes", str(eps), 
                "--generations", str(gen), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        mu, mr = grid_v

        return "%s,%s,mu%i,mr%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        en, 
                                        mu,
                                        mr
                                    )

    experiment_template(
        "mut_grid",
        en,
        "fitness", 
        ["lambda"], 
        [mus[en], mutation_rate[en]], 
        lambda_argument_f, 
        lambda_filename_f

    )
    experiment_template(
        "mut_grid",
        en,
        "novelty", 
        ["lambda"], 
        [mus[en], mutation_rate[en]], 
        lambda_argument_f, 
        lambda_filename_f
    )

def lambda_l_cross_search(en, eps, gen, mu):
    
    def lambda_argument_f(en, grid_v, seed):
        method, lam, cx = grid_v
        return [
                "--enviroment", str(en),
                "--cross_method", str(method),
                "--mutation_rate", str(0),
                "--cross_rate", str(cx),
                "--mu", str(mu),
                "--lambdan", str(lam),
                "--episodes", str(eps), 
                "--generations", str(gen), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        method, l, cx = grid_v

        return "%s,%s,%s,l%i,cx%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        method, 
                                        en, 
                                        l,
                                        cx
                                    )

    experiment_template(
        "cross_grid",
        en,
        "fitness", 
        ["lambda"], 
        [mutation_method, lambdas_[en], cross_rate[en]], 
        lambda_argument_f, 
        lambda_filename_f

    )
    experiment_template(
        "cross_grid",
        en,
        "novelty", 
        ["lambda"], 
        [mutation_method, lambdas_[en], cross_rate[en]], 
        lambda_argument_f, 
        lambda_filename_f
    )


cx_best_lambda = {
    0.3: 20,
    0.5: 20,
    0.7: 30,
    0.9: 20
}

mut_best_mu = {
    0.1: 30,
    0.3: 20,
    0.5: 30,
    0.8: 20
}

def lambda_fit_cross_param_search(en, eps, gen):
    
    def lambda_argument_f(en, grid_v, seed):
        method, cx, mr = grid_v
        return [
                "--enviroment", str(en),
                "--cross_method", str(method),
                "--mutation_rate", str(mr),
                "--cross_rate", str(cx),
                "--mu", str(mut_best_mu[mr]),
                "--lambdan", str(cx_best_lambda[cx]),
                "--episodes", str(eps), 
                "--generations", str(gen), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        method, cx, mr = grid_v

        return "%s,%s,%s,mu%i,mr%.2f,l%i,cx%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        en,
                                        method,
                                        mut_best_mu[mr],
                                        mr,
                                        cx_best_lambda[cx],
                                        cx
                                    )

    experiment_template(
        "cross_param_grid",
        en,
        "fitness", 
        ["lambda"], 
        [mutation_method, cross_rate[en], mutation_rate[en]], 
        lambda_argument_f, 
        lambda_filename_f

    )


cx_best_lambda_n = {
    5: 0.9,
    10: 0.5,
    20: 0.3,
    30: 0.7,
    40: 0.3
}

mut_best_mu_n = {
    5 : 0.1,
    10: 0.5,
    20: 0.8,
    30: 0.8,
    40: 0.8,
}

def lambda_novelty_cross_param_search(en, eps, gen):
    
    def lambda_argument_f(en, grid_v, seed):
        method, lam, mu = grid_v
        return [
                "--enviroment", str(en),
                "--cross_method", str(method),
                "--mutation_rate", str(mut_best_mu_n[mu]),
                "--cross_rate", str(cx_best_lambda_n[lam]),
                "--mu", str(mu),
                "--lambdan", str(lam),
                "--episodes", str(eps), 
                "--generations", str(gen), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        method, l, mu = grid_v

        return "%s,%s,%s,mu%i,mr%.2f,l%i,cx%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        en,
                                        method,
                                        mu,
                                        mut_best_mu_n[mu],
                                        l,
                                        cx_best_lambda_n[l],
                                    )

    experiment_template(
        "cross_param_grid",
        en,
        "novelty", 
        ["lambda"], 
        [mutation_method, lambdas_[en], mus[en]], 
        lambda_argument_f, 
        lambda_filename_f

    )

pops = {
    "cartpole":[1,3,7,15]
}

def diff_population_search(en ):
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
    experiment_template(
        "pop_grid",
        en,
        "fitness", 
        ["lambda"], 
        [mutation_method, mutation_rate, cross_rate], 
        lambda_argument_f, 
        lambda_filename_f

    )
    experiment_template(
        "pop_grid",
        "novelty", 
        ["lambda"], 
        [mutation_method, mutation_rate, cross_rate], 
        lambda_argument_f, 
        lambda_filename_f
    )

mutation_sigma = [ 0.5, 1, 2, 3] #lambda
def lambda_sigma_search(eps):

    def lambda_argument_f(en, grid_v, seed):
        gens, method, sigma = grid_v
        return [
                "--enviroment", str(en),
                "--mutation_sigma", str(sigma),
                "--cross_method", str(method),
                "--episodes", str(eps), 
                "--generations", str(gens), 
                "--seed", str(seed),         
        ]
    
    def lambda_filename_f(en, alg, grid_v):
        gens, method, sigma = grid_v

        return "%s,%s,%s,g%iq,ms%.2f.json" %  (
                                        alg + "_" + str(datetime.datetime.utcnow()),
                                        method,
                                        en, 
                                        gens, 
                                        sigma
                                    )

    experiment_template(
        "sigma_grid",
        "fitness", 
        ["lambda"], 
        [generations, mutation_method, mutation_sigma], 
        lambda_argument_f, 
        lambda_filename_f

    )
    experiment_template(
        "sigma_grid",
        "novelty", 
        ["lambda"], 
        [generations, mutation_method, mutation_sigma], 
        lambda_argument_f, 
        lambda_filename_f
    )




mix_coefs = [0.5, 0.3, 0.1, 0.01, 0.001]
def mix_novelty_grid():
    pass

def archiving_vs_normal():
    pass

def final_showdown():
    pass


if __name__ == "__main__":
    #ep_vs_gen_experiment()
    #enviroments_grid_searches(3, 25)
    #fit_diff(3, 25)
    #lambda_sigma_search(3)
    #lambda_l_cross_search("cartpole", 3, 15, 10)
    #lambda_mu_mut_search("cartpole", 3, 15, 10)
    #lambda_fit_cross_param_search("cartpole", 3, 15)
    lambda_novelty_cross_param_search("cartpole", 3, 15)