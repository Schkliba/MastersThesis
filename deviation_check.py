import itertools
import concurrent.futures
import constants as Cs

SEEDS = [101, 102, 103]
TEST_EVAL_EPS = 5
# lambda fit archving [FrozenTrial(number=80, state=<TrialState.COMPLETE: 1>, values=[118.11550385203829], datetime_start=datetime.datetime(2026, 5, 22, 5, 4, 53, 822616), datetime_complete=datetime.datetime(2026, 5, 22, 5, 36, 8, 359073), params={'crossmethod': 'uniform', 'lambda': 70, 'mu': 70, 'mutation_rate': 0.01, 'cross_rate': 0.9000000000000001, 'sigma': 0.5, 'archiving_period': 4, 'archive_batch': 5, 'cross': 0.6}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.1, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5), 'archiving_period': IntDistribution(high=5, log=False, low=2, step=1), 'archive_batch': IntDistribution(high=5, log=False, low=1, step=1), 'cross': FloatDistribution(high=0.9, log=False, low=0.1, step=0.1)}, trial_id=81, value=None), FrozenTrial(number=86, state=<TrialState.COMPLETE: 1>, values=[118.11550385203829], datetime_start=datetime.datetime(2026, 5, 22, 5, 36, 31, 908497), datetime_complete=datetime.datetime(2026, 5, 22, 6, 11, 5, 128793), params={'crossmethod': 'uniform', 'lambda': 70, 'mu': 70, 'mutation_rate': 0.01, 'cross_rate': 0.9000000000000001, 'sigma': 0.5, 'archiving_period': 4, 'archive_batch': 5, 'cross': 0.6}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.1, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5), 'archiving_period': IntDistribution(high=5, log=False, low=2, step=1), 'archive_batch': IntDistribution(high=5, log=False, low=1, step=1), 'cross': FloatDistribution(high=0.9, log=False, low=0.1, step=0.1)}, trial_id=87, value=None)]
# lambda - this one is bad actually novelty [FrozenTrial(number=4, state=<TrialState.COMPLETE: 1>, values=[-9.291787437438707], datetime_start=datetime.datetime(2026, 5, 19, 22, 25, 13, 886515), datetime_complete=datetime.datetime(2026, 5, 19, 23, 0, 16, 677521), params={'crossmethod': 'mean', 'lambda': 60, 'mu': 60, 'mutation_rate': 0.18, 'cross_rate': 1.0, 'sigma': 2.5}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'crossmethod': CategoricalDistribution(choices=('uniform', 'mean')), 'lambda': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mu': CategoricalDistribution(choices=(40, 50, 60, 70)), 'mutation_rate': FloatDistribution(high=0.5, log=False, low=0.0, step=0.01), 'cross_rate': FloatDistribution(high=1.0, log=False, low=0.3, step=0.1), 'sigma': FloatDistribution(high=3.0, log=False, low=0.5, step=0.5)}, trial_id=208, value=None)]
def task_job(env, alg, args, s):
    df, pop = Cs.ALG_MAPPING[alg].main(args=args)
    print("Testing " + str(s))
    fitnesses = [env.evalutation_b(p, 42, TEST_EVAL_EPS) for p in pop]
    print("Finished seed %d of algorithm %s" % (s, alg))
    return fitnesses

def lambda_deviation(en, method, l, m, cr, mr, dl, dm, dcr, dmr, sigma, n=3):
    #we do not deviate the sigma
    # first we evaluate the current setup
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            print("Launching " + "lambda" + "on Enviroment " + str(en))
            for s in SEEDS:
                env = Cs.ENIVROMENTS[en]()
                args
                future = executor.submit(task_job, alg="lambda", env=en, args=args, s=s)
                stat_futures[future] = s

    # first we try deviate mu and mr
    pass
    #then we adjust 

