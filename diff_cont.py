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
parser.add_argument("-cr", "--cross_rate", help="Rate of crossing individuals", type=float, default=0.6)
parser.add_argument("-mr", "--mutation_rate", help="Rate of mutation", type=float, default=0.9)
parser.add_argument("-p", "--pop", help="Base population", type=int, default=7)
parser.add_argument("-s", "--seed", help="Seed of the random generator", type=int, default=42)

args = parser.parse_args()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", cartpole.CartpolePlayer, fitness=creator.FitnessMax)

ng, l, cr, mr, seed = args.generations, args.pop, args.cross_rate, args.mutation_rate, args.seed
rp = lambda x: cartpole.evalutation(x, seed, 1, True)

toolbox = base.Toolbox()
toolbox.register("evaluate", cartpole.evalutation, seed=seed, episodes=EVAL_EPISODES)
toolbox.register("triOp", cartpole.cartdiff, alpha=mr)
toolbox.register("recombine", cartpole.cartrecomb, prob_filter=cr)
stats = statconf.get_statistics()

alg = containers.DiffAlgContainer(l,toolbox, seed, ng,creator,stats)
alg.replay_f = rp
alg.run()
df = visualisation.logbook2pandas(alg.logbook)
df.to_csv("./Data/diff_fit_"+str(datetime.datetime.utcnow())+".out")
visualisation.single_run_chart(df)
