import containers
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

args = parser.parse_args()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)

ng, l, m, cr, mr, seed = args.generations, args.lambdan, args.mu, args.cross_rate, args.mutation_rate, args.seed
rp = lambda x: cartpole.evalutation(x, seed, 1, True)

toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)
if args.cross_method == "mean":
    toolbox.register("mate", cartpole.cart_mean)
else:
    toolbox.register("mate", cartpole.cart_uniform, prob_filter=args.cross_uni)

toolbox.register("mutate", cartpole.mutcartion, sigma=args.mutation_sigma)

alg = containers.LambdaAlgContainer(l, m, mr, cr, seed, ng,toolbox,creator)
alg.replay_f = rp
alg.run()
df = visualisation.logbook2pandas(alg.logbook)
df.to_csv("./Data/lambda_fit_"+args.cross_method+"_"+"sigma="+str(args.mutation_sigma)+"_"+str(datetime.datetime.utcnow())+".out")
visualisation.single_run_chart(df)
