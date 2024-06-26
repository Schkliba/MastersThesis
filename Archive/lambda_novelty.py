import containers as con
import cartpole
import visualisation
import datetime
import statconf
import argparse
import numpy as np

from deap import base
from deap import creator

EVAL_EPISODES = 10

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--generations", help="Number of generations", type=int, default=10)
parser.add_argument("-cm", "--cross_method", help="Method of crossing individuals", type=str, default="mean")
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.5)
parser.add_argument("-cru", "--cross_uni", help="Method of crossing individuals", type=float, default=0.4)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.01)
parser.add_argument("-ms", "--mutation_sigma", help="Sigma of mutation", type=float, default=1)
parser.add_argument("-l", "--lambdan", help="Base population", type=int, default=30)
parser.add_argument("-m", "--mu", help="Offspring pool", type=int, default=30)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)
parser.add_argument("-p", "--out_path", help="Path to store output data", type=str, default="./Data/Junk/")


args = parser.parse_args()

ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
rp = lambda x: cartpole.evalutation(x, seed, 1, True)

creator.create("FitnessNovelty", base.Fitness, weights=(1.0, ))
#creator.create("Behavior", base.Fitness, weights=(1.0, 1.0)) #end position & angle
creator.create("FitnessTrue", base.Fitness, weights=(1.0, )) 

creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessNovelty , fitness2=creator.FitnessTrue)#, behaviour=creator.Behavior)

toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation_b, seed=seed, episodes=EVAL_EPISODES)
if args.cross_method == "mean":
    toolbox.register("mate", cartpole.cart_mean)
else:
    toolbox.register("mate", cartpole.cart_uniform, prob_filter=args.cross_uni)

toolbox.register("mutate", cartpole.mutcartion, sigma=args.mutation_sigma)

alg = con.LambdaNoveltyAlg(l,m,mr,cr,seed,ng, toolbox, creator)
alg.replay_f = rp
alg.run()
df = visualisation.novelty_logbook2pandas(alg.logbook)
df.to_json(args.out_path + "lambda_nov_"+str(datetime.datetime.utcnow())+".out")
visualisation.single_novelty_run_chart(df)
